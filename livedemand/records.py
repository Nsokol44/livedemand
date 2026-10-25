"""records.py - bitemporal observation store with an as_of(release_time) loader.

Schema (one row per observation version):
  variable      what was measured (sessions, atc, orders, ...)
  unit          unit of the value (count, ...)
  support       spatial support id (e.g. "zip:37996")
  valid_time    the period the observation describes (integer week index)
  release_time  when this version became available (integer week index)
  revision      0 for the first release, 1, 2, ... for later revisions
  value         the observed value
  se            standard error / uncertainty (0 if exact)
  source        provenance tag

This is a profile on bitemporal tables (SQL:2011) and ALFRED-style vintages,
not a new format. Its one job is to make it impossible to read data that did
not exist yet: as_of(t) returns, for every (variable, support, valid_time),
the latest revision whose release_time <= t.
"""
import numpy as np
import pandas as pd

COLUMNS = ["variable", "unit", "support", "valid_time", "release_time",
           "revision", "value", "se", "source"]
KEY = ["variable", "support", "valid_time"]


class LiveStore:
    def __init__(self):
        self._parts = []
        self._df = pd.DataFrame(columns=COLUMNS)
        self._dirty = False

    # ------------------------------------------------------------ writing
    def append_many(self, variable, unit, supports, valid_time, release_time,
                    values, se=0.0, source="sim", revision=0):
        supports = np.asarray(supports)
        n = len(supports)
        if n == 0:
            return
        values = np.asarray(values, dtype=float)
        part = pd.DataFrame({
            "variable": variable, "unit": unit, "support": supports,
            "valid_time": int(valid_time), "release_time": int(release_time),
            "revision": int(revision), "value": values,
            "se": np.broadcast_to(np.asarray(se, dtype=float), (n,)),
            "source": source})
        if int(release_time) < int(valid_time):
            raise ValueError("release_time cannot precede valid_time")
        self._parts.append(part)
        self._dirty = True

    def frame(self):
        if self._dirty:
            self._df = pd.concat([self._df] + self._parts, ignore_index=True)
            self._parts = []
            self._dirty = False
        return self._df

    # ------------------------------------------------------------ reading
    def as_of(self, release_time, variable=None, valid_through=None):
        """Latest revision of each observation released by `release_time`."""
        df = self.frame()
        m = df["release_time"] <= release_time
        if variable is not None:
            m &= df["variable"] == variable
        if valid_through is not None:
            m &= df["valid_time"] <= valid_through
        df = df[m]
        if df.empty:
            return df
        df = df.sort_values(["revision", "release_time"])
        return df.groupby(KEY, sort=False).tail(1).reset_index(drop=True)

    def latest(self, variable=None):
        return self.as_of(np.inf, variable)

    def history(self, variable, support, valid_time):
        df = self.frame()
        m = ((df["variable"] == variable) & (df["support"] == support)
             & (df["valid_time"] == valid_time))
        return df[m].sort_values("release_time").reset_index(drop=True)

    def matrix(self, variable, release_time, support_index, n_time,
               valid_through=None, time_index=None):
        """Dense (n_supports x n_time) array of the as_of(release_time) view.
        Missing observations are 0. support_index maps support id -> row."""
        df = self.as_of(release_time, variable, valid_through)
        out = np.zeros((len(support_index), n_time))
        if df.empty:
            return out
        rows = df["support"].map(support_index).to_numpy()
        vt = df["valid_time"]
        cols = (vt.map(time_index) if time_index else vt).to_numpy().astype(int)
        keep = (cols < n_time) & ~pd.isna(rows)
        out[rows[keep].astype(int), cols[keep]] = df["value"].to_numpy()[keep]
        return out

    def completeness(self, variable, release_time):
        """Estimated first-release completeness: sum(first release) /
        sum(latest version) over weeks already revised as of release_time.
        This is what lets a model correct the freshest, still-incomplete
        week instead of trusting it. Returns 1.0 if nothing revised yet."""
        df = self.as_of(release_time, variable)
        first = self.frame()
        first = first[(first["variable"] == variable) & (first["revision"] == 0)
                      & (first["release_time"] <= release_time)]
        if df.empty or first.empty:
            return 1.0
        merged = first.merge(df[KEY + ["value", "revision"]], on=KEY,
                             suffixes=("_first", "_latest"))
        merged = merged[merged["revision_latest"] > 0]
        if merged.empty or merged["value_latest"].sum() <= 0:
            return 1.0
        return float(merged["value_first"].sum() / merged["value_latest"].sum())

    # ------------------------------------------------------------ io
    def to_csv(self, path):
        self.frame().to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path):
        s = cls()
        s._df = pd.read_csv(path)
        return s
