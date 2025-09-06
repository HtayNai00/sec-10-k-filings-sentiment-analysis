# src/build_1A7_pipeline.py
"""
ONE-STEP PIPELINE:
- Filter tickers to those with dependable Item 1A + Item 7 coverage
- Build a single dataset from data/bronze_v2 focusing on Item 1A + Item 7
  (optionally carry a small slice from Item 1 and Item 7A)

Outputs:
  - <outdir>/dataset.{parquet,csv}  (FINAL single dataset)
  - <outdir>/tickers_filtered.txt   (tickers actually used)
  - <outdir>/coverage_summary.csv   (what passed the filter)

Usage example:
  python src/build_1A7_pipeline.py ^
    --report logs/parse_report_v2.csv ^
    --bronze data/bronze_v2 ^
    --years 2020 2021 2022 2023 2024 ^
    --min-years 4 ^
    --carry-1 800 ^
    --carry-7a 800 ^
    --outdir data/silver_1A7
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
import pandas as pd

# -------- helpers for the section extraction --------
SECTION_COL_CANDIDATES = ["section","item","header"]
TEXT_COL_CANDIDATES    = ["text","content","body"]
SRC_COL_CANDIDATES     = ["source_doc","source","doc","filename"]

def _norm_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s.\-:_]+", " ", s)
    if "item 1a" in s or s in {"1a","item1a"}: return "1A"
    if "item 1"  in s or s in {"1","item1"}:   return "1"
    if "item 7a" in s or s in {"7a","item7a"}: return "7A"
    if "item 7"  in s or s in {"7","item7"}:   return "7"
    return s

def _pick_column(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns: return c
    return None

def _first_n(s: str, n: int) -> str:
    if not isinstance(s, str) or n <= 0: return ""
    return s[:n]

def _join(parts, sep="\n\n"):
    return sep.join([p for p in parts if p])

def _extract_sections(one_parquet: Path, carry1: int, carry7a: int, min_chars: int) -> list[dict]:
    df = pd.read_parquet(one_parquet)

    sec_col = _pick_column(df, SECTION_COL_CANDIDATES)
    txt_col = _pick_column(df, TEXT_COL_CANDIDATES)
    src_col = _pick_column(df, SRC_COL_CANDIDATES)
    if not sec_col or not txt_col: return []

    df["_norm"] = df[sec_col].map(_norm_label)
    sec_map = {}
    for key in ["1","1A","7","7A"]:
        sub = df[df["_norm"] == key]
        if len(sub):
            row = sub.loc[sub[txt_col].astype(str).str.len().idxmax()]
            sec_map[key] = dict(text=str(row[txt_col]), src=str(row.get(src_col, "")))
        else:
            sec_map[key] = None

    carry_head = _first_n(sec_map["1"]["text"], carry1)   if sec_map["1"]  else ""
    carry_tail = _first_n(sec_map["7A"]["text"], carry7a) if sec_map["7A"] else ""

    out = []
    for key in ["1A","7"]:
        if not sec_map[key]: continue
        main = sec_map[key]["text"]
        composed = _join([carry_head, main, carry_tail])
        if len(composed) < min_chars:
            composed = main
        out.append({
            "section": key,
            "text": composed,
            "text_len": len(composed),
            "src": sec_map[key]["src"],
            "has_1":  int(sec_map["1"]  is not None),
            "has_1A": int(sec_map["1A"] is not None),
            "has_7":  int(sec_map["7"]  is not None),
            "has_7A": int(sec_map["7A"] is not None),
        })
    return out

# -------- coverage filtering (from parse report) --------
def filter_tickers(report_csv: Path, years: list[int] | None, min_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(report_csv)
    needed = {"ticker","year","item_1A","item_7","status"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in {report_csv}: {sorted(missing)}")
        sys.exit(1)

    if years:
        df = df[df["year"].isin(years)].copy()

    df = df[df["status"] == "OK"].copy()
    df["ok_1a7"] = (df["item_1A"] == 1) & (df["item_7"] == 1)

    summary = (
        df.groupby("ticker")
          .agg(years_total=("year","nunique"), years_ok_1a7=("ok_1a7","sum"))
          .reset_index()
          .sort_values(["years_ok_1a7","ticker"], ascending=[False, True])
    )

    keep = summary[summary["years_ok_1a7"] >= min_years]["ticker"].sort_values().to_frame()
    return keep, summary

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report",   required=True, help="logs/parse_report_v2.csv")
    ap.add_argument("--bronze",   default="data/bronze_v2", help="Root of sections.parquet tree")
    ap.add_argument("--years",    nargs="*", type=int, help="Restrict years (default: all years present)")
    ap.add_argument("--min-years", type=int, default=4, help="Require at least this many years with both 1A & 7")
    ap.add_argument("--carry-1",  type=int, default=800, help="Max chars to carry from Item 1 (prepend)")
    ap.add_argument("--carry-7a", type=int, default=800, help="Max chars to carry from Item 7A (append)")
    ap.add_argument("--min-chars", type=int, default=800, help="Drop rows shorter than this")
    ap.add_argument("--outdir",   default="data/silver_1A7", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) pick tickers that pass the 1A+7 coverage rule
    keep_df, summary = filter_tickers(Path(args.report), args.years, args.min_years)
    if keep_df.empty:
        print("[WARN] No tickers passed the filter. Exiting.")
        return
    tickers = keep_df["ticker"].tolist()

    # save artifacts
    (outdir / "tickers_filtered.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")
    summary.to_csv(outdir / "coverage_summary.csv", index=False)
    print(f"[OK] {len(tickers)} tickers → {outdir/'tickers_filtered.txt'}")
    print(f"[OK] Coverage summary → {outdir/'coverage_summary.csv'}")

    # 2) build ONE final dataset from bronze_v2 for those tickers/years
    rows = []
    bronzeroot = Path(args.bronze)

    for t in tickers:
        t_dir = bronzeroot / t
        if not t_dir.exists(): continue
        years = [str(y) for y in args.years] if args.years else \
                sorted([d.name for d in t_dir.iterdir() if d.is_dir() and d.name.isdigit()])

        for y in years:
            p = t_dir / y / "sections.parquet"
            if not p.exists(): continue
            section_rows = _extract_sections(p, args.carry_1, args.carry_7a, args.min_chars)
            for r in section_rows:
                r.update({"ticker": t, "year": int(y)})
                rows.append(r)

    if not rows:
        print("[WARN] No section rows built. Check --years, bronze tree, or carries.")
        return

    out = pd.DataFrame(rows).sort_values(["ticker","year","section"]).reset_index(drop=True)
    out_parquet = outdir / "dataset.parquet"
    out_csv     = outdir / "dataset.csv"
    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Final dataset: {len(out)} rows → {out_parquet}")
    print(f"[OK] Final dataset: {len(out)} rows → {out_csv}")

if __name__ == "__main__":
    main()
