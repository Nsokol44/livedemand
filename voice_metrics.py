"""voice_metrics.py - before/after measurements for the L9 voice pass.

Counting rules (applied identically before and after):
  * Document = manuscript.md. The References section is excluded from
    the voice / spelling / sentence metrics (reference titles and quoted
    matter are never altered); em dashes are counted both ways.
  * Prose = body lines that are not headings, tables, images, display
    math, or blank lines.
  * "we/our" = case-insensitive whole-word matches of we|our in prose;
    "this paper/the paper" = case-insensitive matches of the two phrases
    in prose.
  * British tokens = matches of the curated British-form wordlist below
    in prose (American false friends such as "advertising", "four",
    "revised", "improvising" excluded by construction of the list).
  * Sentences = prose split on [.!?] followed by whitespace + capital /
    digit / opening quote; length in whitespace-delimited tokens with
    markdown emphasis stripped. Mean and share > 40 words reported.

Usage: python voice_metrics.py [path-to-manuscript.md]
"""
import re
import sys

BRITISH = [
    "anonymised", "behaviour", "centre", "defence", "formalising",
    "generalised", "labelled", "labelling", "modelled", "modelling",
    "neighbour", "neighbours", "neighbourhood", "neighbouring",
    "penalised", "practise", "programme", "randomisation",
    "randomised", "realisation", "realised", "standardised",
    "standardising", "summarised", "characterised", "organised",
    "recognised", "analysed", "minimised", "emphasised",
    "specialised", "utilised", "finalised", "fibre", "licence",
    "whilst", "amongst", "towards",
]
BRITISH_RE = re.compile(r"\b(" + "|".join(BRITISH) + r")\b", re.I)


def prose_lines(text):
    out = []
    in_math = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$$"):
            in_math = not in_math
            continue
        if in_math or not s or s.startswith("#") or s.startswith("|") \
                or s.startswith("!["):
            continue
        out.append(s)
    return out


def main(path):
    text = open(path).read()
    body = text.split("## References")[0]
    prose = "\n".join(prose_lines(body))
    plain = re.sub(r"[*_`$]", "", prose)
    plain = re.sub(r"\\[a-zA-Z]+", " ", plain)

    em_all = text.count("\u2014")
    em_body = body.count("\u2014")
    we = len(re.findall(r"\b(we|our)\b", plain, flags=re.I))
    tp = len(re.findall(r"this paper", plain, flags=re.I))
    thp = len(re.findall(r"the paper", plain, flags=re.I))
    brit = BRITISH_RE.findall(plain)
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", plain)
             if len(s.split()) >= 3]
    lens = [len(s.split()) for s in sents]
    mean_len = sum(lens) / len(lens)
    over40 = sum(1 for L in lens if L > 40)

    words_body = len(body.split())
    words_all = len(text.split())
    print(f"file: {path}")
    print(f"words (whole md incl. references/markup): {words_all}")
    print(f"words (body, before References):          {words_body}")
    print(f"em dashes: whole={em_all}  body={em_body}")
    print(f"voice: we/our={we}   'this paper'={tp}  'the paper'={thp}"
          f"  (paper-phrases total={tp + thp})")
    print(f"British prose tokens: {len(brit)}")
    print(f"sentences: n={len(lens)}  mean={mean_len:.1f} words  "
          f">40 words: {over40}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../manuscript.md")
