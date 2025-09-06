# src/diagnose_candidates_v2.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
from textwrap import indent

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from parse_filings_v2 import (
    _score_doc, read_text_from_doc, auto_pick_html,
    HEADER_PATTERNS, SPECIFIC_ENDERS_COMPILED, ANY_ITEM_PATTERN,
    find_candidate_starts, compute_span_with_ender, content_score, is_in_toc_zone
)

ANCHORS = {
    "Item 1":  [r"(?i)\bBusiness\b", r"(?i)\bOverview\b"],
    "Item 1A": [r"(?i)Risk Factors", r"(?i)material risks", r"(?i)could adversely affect"],
    "Item 7":  [r"(?i)Management.?s Discussion", r"(?i)Results of Operations", r"(?i)Liquidity and Capital Resources", r"(?i)Critical Accounting"],
    "Item 7A": [r"(?i)Quantitative and Qualitative", r"(?i)Market Risk", r"(?i)Interest Rate Risk", r"(?i)Foreign Currency", r"(?i)Commodity Price"],
}

def summarize_text(snippet: str, n: int = 240) -> str:
    snippet = re.sub(r"\s+", " ", snippet.strip())
    return (snippet[:n] + "…") if len(snippet) > n else snippet

def _print_ranked_candidates(label: str, text: str, filtered_matches, ender_pat, show_preview: bool):
    cand_info = []
    for m in filtered_matches:
        s_idx = m.start()
        span = compute_span_with_ender(text, s_idx, ender_pat)
        cscore = content_score(text, span[0], span[1], ANCHORS[label], label)
        cand_info.append((s_idx, span, cscore))

    cand_info.sort(key=lambda x: (x[2], x[1][1] - x[1][0]), reverse=True)

    for rank, (s_idx, (s, e), sc) in enumerate(cand_info, 1):
        seg = summarize_text(text[s: min(e, s + 300)])
        print(f" {rank:>2}. [Primary] start={s_idx:,}  span_len={e - s:,}  content_score={sc}  preview='{seg}'")

    best_s, best_e = cand_info[0][1]
    print(f"→ best {label} span Len = {best_e - best_s:,}")
    if show_preview:
        chunk = text[best_s: min(best_e, best_s + 1200)]
        print(indent(chunk, "    "))

def main():
    ap = argparse.ArgumentParser(description="V2 Diagnose sections (1,1A,7,7A) and chosen HTML.")
    ap.add_argument("--raw-base", default="data/raw")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    year_dir = Path(args.raw_base) / args.ticker / str(args.year)
    if not year_dir.exists():
        raise SystemExit(f"Folder not found: {year_dir}")

    cands = list(year_dir.glob("*.htm")) + list(year_dir.glob("*.html")) \
          + list(year_dir.glob("*.HTM")) + list(year_dir.glob("*.HTML"))
    if not cands:
        raise SystemExit(f"No HTML files in {year_dir}")

    scored = []
    for p in cands:
        sc, metrics = _score_doc(p)
        scored.append((sc, p, metrics))
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n=== [{args.ticker} {args.year}] HTML candidate scores (top {args.top_k}) ===")
    for i, (sc, p, m) in enumerate(scored[: args.top_k], 1):
        print(f"{i:>2}. score={sc:>6}  name={p.name}  size={m.get('size')}  text_len={m.get('text_len')}  header_hits={m.get('header_hits')}")

    picked = auto_pick_html(year_dir)
    print(f"\n→ auto_pick_html would choose: {picked.name}")

    text = read_text_from_doc(picked)
    print(f"\nChosen doc text length: {len(text):,}")

    labels = ["Item 1", "Item 1A", "Item 7", "Item 7A"] if args.all else ["Item 1A", "Item 7", "Item 7A"]
    for label in labels:
        print(f"\n--- Diagnose {label} ---")
        lab_pat = re.compile(HEADER_PATTERNS[label])
        ender_pat = SPECIFIC_ENDERS_COMPILED.get(label)

        starts = [m for m in lab_pat.finditer(text)]
        print(f"Found {len(starts)} raw header matches (before filters).")

        filtered = find_candidate_starts(text, label, HEADER_PATTERNS[label])
        print(f"{len(filtered)} candidates after primary filters.")
        if filtered:
            _print_ranked_candidates(label, text, filtered, ender_pat, args.preview)
            continue

        # Relaxed passes
        if label == "Item 1":
            relaxed = [m for m in lab_pat.finditer(text) if not is_in_toc_zone(text, m.start())]
            print(f"{len(relaxed)} candidates after relaxed Item 1 filter.")
            if relaxed:
                _print_ranked_candidates(label, text, relaxed, ender_pat, args.preview)
                continue

        if label == "Item 7A":
            relaxed = [
                m for m in lab_pat.finditer(text)
                if not is_in_toc_zone(text, m.start())
                and not ANY_ITEM_PATTERN.search(text, m.start()+1, min(len(text), m.start()+400))
            ]
            print(f"{len(relaxed)} candidates after relaxed Item 7A filter.")
            if relaxed:
                _print_ranked_candidates(label, text, relaxed, ender_pat, args.preview)
                continue

            # Alt-title pass
            ALT_7A_TITLES = [
                r"(?i)\bQuantitative\s+and\s+Qualitative\s+Disclosure[s]?\s+About\s+Market\s+Risk\b",
                r"(?i)\bQuantitative\s+and\s+Qualitative\s+Disclosures?\s+About\s+Market\s+Risks?\b",
                r"(?i)\bMarket\s+Risk[s]?\s+Disclosures?\b",
            ]
            alt_matches = []
            for pat in ALT_7A_TITLES:
                alt_matches.extend(list(re.finditer(pat, text)))
            alt_matches = [
                m for m in alt_matches
                if not is_in_toc_zone(text, m.start())
                and not ANY_ITEM_PATTERN.search(text, m.start()+1, min(len(text), m.start()+400))
            ]
            print(f"{len(alt_matches)} candidates from 7A alt-title search.")
            if alt_matches:
                _print_ranked_candidates(label, text, alt_matches, ender_pat, args.preview)
                continue

        print("No viable candidates.")

if __name__ == "__main__":
    main()
