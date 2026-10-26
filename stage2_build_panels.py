#!/usr/bin/env python3
"""stage2_build_panels.py - build the stage-2 real-data panels and the
calibration targets from the archived raw data.

Inputs (archived under ../../data/raw/, see data/raw/MANIFEST.md; never
re-fetched):
  TTS_LBNL_public_file_28-Jul-2026_all.csv   Tracking the Sun (LBNL) 2026
  ca_dmv/dmv_snapshot_YYYY-01-01.csv         CA DMV fuel-type counts by ZIP
  acs2023_b11001_zcta_source.dat             ACS 2023 5-yr, households
  acs2023_b25003_zcta_source.dat             ACS 2023 5-yr, housing units
  2020_Gaz_zcta_national.txt                 2020 Gazetteer ZCTA centroids

Outputs (code/results/):
  stage2_solar_panel.csv        zip x year installs (zero-filled), households,
                                housing units, centroid, cumulative installs
  stage2_ev_panel.csv           zip x year EV stock (BEV+PHEV) and net additions
  stage2_ca_households.csv      empirical household distribution (calibration)
  stage2_calibration_targets.csv  observed quantities the simulator is matched to
  stage2_calibration_params.json  chosen simulator parameters + match record

Deterministic: no randomness except the fixed-seed Moran's-I matching draws
(seed 20261025). One-command rerunnable: python stage2_build_panels.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "data", "raw")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from run_realgeo import knn_W, band_W  # the paper's own graph builders

SEED = 20261025
PANEL_YEARS = list(range(2005, 2025))          # zero-filled solar panel years
FIT_YEARS = list(range(2010, 2025))            # demonstration fit window
RES_SEGMENTS = {"RES", "RES_SF", "RES_MF"}


def morans_I(x, W):
    x = np.asarray(x, float)
    z = x - x.mean()
    Wd = W.toarray() if sparse.issparse(W) else W
    s0 = Wd.sum()
    num = float(z @ (Wd @ z))
    return float(len(x) / s0 * num / (z @ z))


def planar(lat, lon):
    return np.column_stack([lon * np.cos(np.radians(lat.mean())), lat])


def zip_column(path):
    """DMV schema drift: 2019-2023 files head the ZIP column 'Zip Code',
    2024-2026 files 'ZIP Code'."""
    cols = pd.read_csv(path, nrows=0).columns
    return "ZIP Code" if "ZIP Code" in cols else "Zip Code"



def norm_zip(s):
    """Normalise a raw ZIP string: some TTS providers write '95648.0';
    DMV and TTS both carry occasional ZIP+4 and '-1' missing codes."""
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s[:5] if len(s) >= 5 and s[:5].isdigit() else s


def scan_tts_ca():
    """One streaming pass over the TTS file: counts of CA residential
    installs by (zip, year), the CA zip set, and record totals."""
    counts = {}
    ca_zips = set()
    total = 0
    path = os.path.join(RAW, "TTS_LBNL_public_file_28-Jul-2026_all.csv")
    cols = ["zip_code", "installation_date", "state", "customer_segment"]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000,
                             dtype=str, low_memory=False):
        c = chunk[(chunk["state"] == "CA")
                  & chunk["customer_segment"].isin(RES_SEGMENTS)]
        total += len(c)
        c = c.assign(zip=c["zip_code"].map(norm_zip),
                     year=pd.to_numeric(
                         c["installation_date"].str.slice(0, 4),
                         errors="coerce"))
        ca_zips |= set(c["zip"])
        c = c[c["year"].between(2000, 2025)]
        for (z, y), v in c.groupby(["zip", "year"]).size().items():
            counts[(z, int(y))] = counts.get((z, int(y)), 0) + int(v)
    return counts, ca_zips, total


def build_universe(ca_zips):
    """CA ZCTA universe: ZIPs evidenced as Californian in BOTH the CA DMV
    snapshots and the TTS state field (the DMV file alone carries
    out-of-state mailing ZIPs), with a 2020 Gazetteer centroid and a
    positive ACS household count."""
    dmv_zips = set()
    for yr in range(2019, 2027):
        p = os.path.join(RAW, "ca_dmv", f"dmv_snapshot_{yr}-01-01.csv")
        zcol = zip_column(p)
        z = pd.read_csv(p, usecols=[zcol], dtype=str)[zcol]
        dmv_zips |= set(z.map(norm_zip))
    gaz = pd.read_csv(os.path.join(RAW, "2020_Gaz_zcta_national.txt"),
                      sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["zip"] = gaz["GEOID"].str.strip().str.zfill(5)
    keep = dmv_zips & set(ca_zips)
    gaz = gaz[gaz["zip"].isin(keep)][["zip", "INTPTLAT", "INTPTLONG"]]
    gaz.columns = ["zip", "lat", "lon"]

    def acs_total(path, var):
        df = pd.read_csv(path, sep="|", usecols=["GEO_ID", var], dtype=str)
        df = df[df["GEO_ID"].str.startswith("860")]  # ZCTA summary level
        df["zip"] = df["GEO_ID"].str[-5:]
        return df[["zip", var]]

    hh = acs_total(os.path.join(RAW, "acs2023_b11001_zcta_source.dat"),
                   "B11001_E001").rename(columns={"B11001_E001": "households"})
    hu = acs_total(os.path.join(RAW, "acs2023_b25003_zcta_source.dat"),
                   "B25003_E001").rename(columns={"B25003_E001": "housing_units"})
    uni = gaz.merge(hh, on="zip").merge(hu, on="zip")
    uni["households"] = pd.to_numeric(uni["households"], errors="coerce")
    uni["housing_units"] = pd.to_numeric(uni["housing_units"], errors="coerce")
    uni = uni[(uni["households"] > 0) & uni["lat"].notna()]
    return uni.sort_values("zip").reset_index(drop=True)


def build_solar_panel(uni, counts, total_ca_res):
    kept = int(sum(v for (z, y), v in counts.items() if z in set(uni["zip"])))
    meta = uni.set_index("zip")
    rows = []
    for z in uni["zip"]:
        u = meta.loc[z]
        cum = 0
        for y in range(2000, 2025):
            n_y = counts.get((z, y), 0)
            if y in PANEL_YEARS:
                rows.append(dict(zip=z, year=y, installs=n_y, cum_prev=cum,
                                 households=int(u["households"]),
                                 housing_units=int(u["housing_units"]),
                                 lat=float(u["lat"]), lon=float(u["lon"])))
            cum += n_y
    panel = pd.DataFrame(rows)
    return panel, dict(ca_residential_records=int(total_ca_res),
                       kept_in_universe=int(kept))


def build_ev_panel(uni):
    zips = list(uni["zip"])
    zset = set(zips)
    stocks = {}
    for yr in range(2019, 2027):
        p = os.path.join(RAW, "ca_dmv", f"dmv_snapshot_{yr}-01-01.csv")
        tot = pd.Series(0.0, index=zips)
        zcol = zip_column(p)
        for chunk in pd.read_csv(p, chunksize=500_000,
                                 dtype={zcol: str}):
            c = chunk[chunk["Fuel"].isin(["Battery Electric",
                                          "Plug-in Hybrid"])]
            g = c.groupby(c[zcol].map(norm_zip))["Vehicles"].sum()
            g = g[g.index.isin(zset)]
            tot.loc[g.index] += g.to_numpy()
        stocks[yr] = tot
    meta = uni.set_index("zip")
    rows = []
    for z in zips:
        for yr in range(2020, 2027):
            prev = float(stocks[yr - 1][z])
            rows.append(dict(zip=z, year=yr, ev_stock=float(stocks[yr][z]),
                             ev_stock_prev=prev,
                             net_additions=float(stocks[yr][z]) - prev,
                             households=int(meta.loc[z, "households"]),
                             lat=float(meta.loc[z, "lat"]),
                             lon=float(meta.loc[z, "lon"])))
    return pd.DataFrame(rows)


def calibration_targets(uni, solar):
    fit = solar[solar["year"].isin(FIT_YEARS)]
    piv = fit.pivot(index="zip", columns="year", values="installs").fillna(0)
    piv = piv.reindex(uni["zip"]).fillna(0)
    hh = uni.set_index("zip")["households"]
    total = piv.sum(axis=1)
    a = np.log((total.to_numpy() + 0.5) / hh.to_numpy() * 1000.0)
    xy = planar(uni["lat"].to_numpy(), uni["lon"].to_numpy())
    latlon = uni[["lat", "lon"]].to_numpy()
    graphs = {"k3": knn_W(xy, 3), "k6": knn_W(xy, 6), "k10": knn_W(xy, 10),
              "band100km": band_W(latlon, 100.0)[0]}
    rows = [dict(quantity="panel ZCTAs", value=len(uni)),
            dict(quantity="CA residential installs in panel, 2005-2024",
                 value=int(solar["installs"].sum())),
            dict(quantity="CA residential installs in fit window, 2010-2024",
                 value=int(fit["installs"].sum())),
            dict(quantity="median households per ZCTA",
                 value=float(hh.median())),
            dict(quantity="mean annual installs per 1000 households (fit window)",
                 value=float((fit["installs"] / fit["households"] * 1000).mean())),
            dict(quantity="receptivity proxy: sd of log installs per 1000 HH",
                 value=float(a.std(ddof=1))),
            dict(quantity="receptivity proxy: IQR of log installs per 1000 HH",
                 value=float(np.quantile(a, 0.75) - np.quantile(a, 0.25)))]
    for name, W in graphs.items():
        rows.append(dict(quantity=f"Moran's I of receptivity proxy, {name}",
                         value=morans_I(a, W)))
    cum_rate = np.log1p(total.to_numpy() / hh.to_numpy() * 1000.0)
    rows.append(dict(quantity="Moran's I of log1p cumulative adoption rate, k6",
                     value=morans_I(cum_rate, graphs["k6"])))
    return pd.DataFrame(rows), a


def match_simulator_params(a_sd, moran_k6):
    """Choose rho_u by matching Moran's I of the simulator's spatial
    receptivity component on the study's real-geometry k=6 graph; then
    sigma_u so total receptivity spread matches the observed proxy, with
    the covariate share held at the paper's beta_sd = 0.4."""
    from livedemand.world import load_real_xy
    coords = os.path.join(HERE, "data", "zcta_sample_1500.csv")
    xy = load_real_xy(coords, 1500)
    W = knn_W(xy, 6)
    n = W.shape[0]
    I = sparse.identity(n, format="csc")
    rng = np.random.default_rng(SEED)
    draws = [rng.standard_normal(n) for _ in range(25)]
    best = None
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        inv = spla.splu((I - rho * W).tocsc())
        m = float(np.mean([morans_I(inv.solve(z), W) for z in draws]))
        if best is None or abs(m - moran_k6) < abs(best[1] - moran_k6):
            best = (rho, m)
    rho_u, moran_sim = best
    inv_u = spla.splu((I - rho_u * W).tocsc())
    var_u_factor = float(np.var(inv_u.solve(draws[0])))   # at sigma_u = 1
    inv_x = spla.splu((I - 0.6 * W).tocsc())
    Z = np.column_stack([inv_x.solve(rng.standard_normal(n)) for _ in range(4)])
    Z = (Z - Z.mean(0)) / Z.std(0)
    beta = rng.normal(0, 0.4, 4)
    var_x = float(np.var(Z @ beta))                        # at beta_sd = 0.4
    share_u = var_u_factor / (var_u_factor + var_x)
    sigma_u = float(a_sd * np.sqrt(share_u) / np.sqrt(var_u_factor))
    return dict(rho_u=float(rho_u), moran_sim_at_rho_u=moran_sim,
                moran_observed=float(moran_k6), sigma_u=sigma_u, beta_sd=0.4,
                var_factor_spatial=var_u_factor, var_covariate=var_x,
                receptivity_sd_target=float(a_sd))


def main():
    counts, ca_zips, total_ca_res = scan_tts_ca()
    uni = build_universe(ca_zips)
    solar, coverage = build_solar_panel(uni, counts, total_ca_res)
    solar.to_csv(os.path.join(OUT, "stage2_solar_panel.csv"), index=False)
    ev = build_ev_panel(uni)
    ev.to_csv(os.path.join(OUT, "stage2_ev_panel.csv"), index=False)
    uni[["zip", "households"]].to_csv(
        os.path.join(OUT, "stage2_ca_households.csv"), index=False)
    targets, a = calibration_targets(uni, solar)
    targets.to_csv(os.path.join(OUT, "stage2_calibration_targets.csv"),
                   index=False)
    moran_k6 = float(targets.loc[
        targets["quantity"] == "Moran's I of receptivity proxy, k6",
        "value"].iloc[0])
    params = match_simulator_params(float(a.std(ddof=1)), moran_k6)
    params.update(coverage)
    with open(os.path.join(OUT, "stage2_calibration_params.json"), "w") as fh:
        json.dump(params, fh, indent=1)
    print(targets.to_string(index=False))
    print(json.dumps(params, indent=1))
    print("panels built:", len(uni), "ZCTAs;",
          int(solar["installs"].sum()), "solar installs;",
          int(ev[ev["year"] == 2026]["ev_stock"].sum()),
          "EVs at 2026-01-01")


if __name__ == "__main__":
    main()
