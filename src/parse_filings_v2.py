# src/parse_filings_v2.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple, Optional
import warnings
from html import unescape

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd

# ------------------ Header detection ------------------
HEADER_PATTERNS = {
    "Item 1":  r"(?i)\bItem[\s\xa0]*1[\s\xa0]*\.?\b(?!A\b)",   # avoid 1A
    "Item 1A": r"(?i)\bItem[\s\xa0]*1A[\s\xa0]*\.?\b",
    "Item 7":  r"(?i)\bItem[\s\xa0]*7[\s\xa0]*\.?\b(?!A\b)",   # avoid 7A
    "Item 7A": r"(?i)\bItem[\s\xa0]*7A[\s\xa0]*\.?\b",
}

ANY_ITEM_PATTERN = re.compile(r"(?i)\bItem[\s\xa0]*[0-9]{1,2}[A-C]?\s*\.?\b")

SPECIFIC_ENDERS = {
    "Item 1":  r"(?i)\bItem[\s\xa0]*1A\b|\bItem[\s\xa0]*1B\b|\bItem[\s\xa0]*2\b",
    "Item 1A": r"(?i)\bItem[\s\xa0]*1B\b|\bItem[\s\xa0]*2\b",
    "Item 7":  r"(?i)\bItem[\s\xa0]*7A\b|\bItem[\s\xa0]*8\b",
    "Item 7A": r"(?i)\bItem[\s\xa0]*8\b",
}
SPECIFIC_ENDERS_COMPILED = {k: re.compile(v) for k, v in SPECIFIC_ENDERS.items()}

# Minimum character gates
MIN_CHARS = {
    "Item 1":  10000,
    "Item 1A": 8000,
    "Item 7":  10000,
    "Item 7A": 3000,
}

# TOC detection
TOC_PAT = re.compile(r"(?i)\bTable of Contents\b")

def is_in_toc_zone(text: str, idx: int) -> bool:
    doc_len = len(text)
    early = doc_len and idx < int(doc_len * 0.05)
    s = max(0, idx - 250)
    e = min(doc_len, idx + 250)
    near_toc = bool(TOC_PAT.search(text[s:e]))
    return early and near_toc

# --- Content/anchor heuristics ---
ANCHORS_1 = [
    r"(?i)\bBusiness\b",
    r"(?i)\bOverview\b",
]
ANCHORS_7 = [
    r"(?i)Management.?s Discussion",
    r"(?i)Results of Operations",
    r"(?i)Liquidity and Capital Resources",
    r"(?i)Critical Accounting",
]
ANCHORS_7A = [
    r"(?i)Quantitative and Qualitative",
    r"(?i)Market Risk",
    r"(?i)Interest Rate Risk",
    r"(?i)Foreign Currency",
    r"(?i)Commodity Price",
]
ANCHORS_1A = [
    r"(?i)Risk Factors",
    r"(?i)material risks",
    r"(?i)could adversely affect",
]

ALT_7A_TITLES = [
    r"(?i)\bQuantitative\s+and\s+Qualitative\s+Disclosure[s]?\s+About\s+Market\s+Risk\b",
    r"(?i)\bQuantitative\s+and\s+Qualitative\s+Disclosures?\s+About\s+Market\s+Risks?\b",
    r"(?i)\bMarket\s+Risk[s]?\s+Disclosures?\b",
]
ALT_7A_TITLES_COMPILED = [re.compile(p) for p in ALT_7A_TITLES]

RELATING_PAT = re.compile(r"(?i)\b(relating to|related to|with respect to)\b")

def has_nearby_item_after(text: str, start_idx: int, radius: int = 400) -> bool:
    end = min(len(text), start_idx + radius)
    return bool(ANY_ITEM_PATTERN.search(text, start_idx + 1, end))

CROSS_REF_PAT = re.compile(r"(?i)\b(see|refer to|as discussed in)\b|(?:of|in)\s+Part\s+[IVXLC]+\b")

def is_cross_reference_use(text: str, start_idx: int, lookahead: int = 220) -> bool:
    end = min(len(text), start_idx + lookahead)
    return bool(CROSS_REF_PAT.search(text[start_idx:end]))

def content_score(text: str, start: int, end: int, anchors: List[str], label: str) -> int:
    win_end = min(len(text), start + 6000, end)
    window = text[start:win_end]
    score = len(window) // 50  # base contentfulness

    # Anchor boosts
    for pat in anchors:
        if re.search(pat, window):
            score += 200

    if label == "Item 1A":
        if re.search(r"(?i)\bRisk\s+Factors\b", window[:300]):
            score += 400

    if label == "Item 1":
        # Strongly prefer an opening that looks like "Item 1 ... Business"
        if re.search(r"(?i)\bItem[\s\xa0]*1\b.{0,120}\bBusiness\b", window[:300], flags=0):
            score += 500
        # If "Business" isn't seen early, demote
        if not re.search(r"(?i)\bBusiness\b", window[:600]):
            score -= 500
        # Officer bios are a common short stub
        if re.search(r"(?i)\bexecutive officers?\b", window[:800]):
            score -= 200

    if has_nearby_item_after(text, start, radius=400):
        score -= 500
    if is_cross_reference_use(text, start, lookahead=220):
        score -= 600

    next_100 = text[start:start+120]
    nl_pos = next_100.find("\n")
    if 0 <= nl_pos <= 80:
        score += 120

    return score

# ------------------ Document selection ------------------
GOOD_TERMS = [r"(?i)\bFORM\s*10[-\s]*K\b", r"(?i)\bANNUAL\s+REPORT\b"]
TARGET_HEADERS = [
    r"(?i)\bItem[\s\xa0]*1\b(?!A\b)",
    r"(?i)\bItem[\s\xa0]*1A\b",
    r"(?i)\bItem[\s\xa0]*7[\s\xa0]*\.?(?!A\b)",
    r"(?i)\bItem[\s\xa0]*7A\b",
]
BAD_NAME_HINTS = ["index", "cover", "toc", "table", "summary", "xbrl", "xml", "exhibit", "wrap", "presentation", "press"]

def _read_plain_text_for_scoring(p: Path) -> str:
    raw = p.read_text(encoding="utf-8", errors="ignore")
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _score_doc(p: Path) -> tuple[int, dict]:
    try:
        txt = _read_plain_text_for_scoring(p)
    except Exception:
        return (-10**9, {"error": "read-failed", "size": p.stat().st_size, "name": p.name})

    size = p.stat().st_size
    score = 0
    score += min(size // 1024, 5000)
    score += min(len(txt) // 1000, 20000)

    for pat in GOOD_TERMS:
        if re.search(pat, txt):
            score += 10000

    header_hits = 0
    for pat in TARGET_HEADERS:
        header_hits += len(re.findall(pat, txt))
    score += header_hits * 1500

    lower_name = p.name.lower()
    for hint in BAD_NAME_HINTS:
        if hint in lower_name:
            score -= 3000

    if len(txt) < 10000:
        score -= 5000
    if header_hits == 0:
        score -= 3000

    metrics = {"size": size, "text_len": len(txt), "header_hits": header_hits, "name": p.name}
    return (score, metrics)

def auto_pick_html(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    cands = list(path.glob("*.htm")) + list(path.glob("*.html")) + list(path.glob("*.HTM")) + list(path.glob("*.HTML"))
    if not cands:
        raise FileNotFoundError(f"No .htm/.html in {path}")

    scored = []
    for p in cands:
        sc, m = _score_doc(p)
        scored.append((sc, p, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ------------------ Parse HTML/XML to text ------------------
def looks_like_xml(s: str) -> bool:
    if s.lstrip().startswith("<?xml"):
        return True
    if re.search(r"<(xbrli:|xbrl[ >])", s, flags=re.I):
        return True
    if re.search(r"</?DOCUMENT[^>]*type=['\"]XML['\"]", s, flags=re.I):
        return True
    return False

def read_text_from_doc(doc_path: Path) -> str:
    raw = doc_path.read_text(encoding="utf-8", errors="ignore")
    parser = "xml" if looks_like_xml(raw) else "lxml"
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(raw, parser)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ------------------ Section extraction ------------------
def find_candidate_starts(text: str, label: str, label_regex: str) -> List[re.Match]:
    lab_pat = re.compile(label_regex)
    raw = [m for m in lab_pat.finditer(text)]

    # Always drop TOC/cover and index-like rows
    cands = [
        m for m in raw
        if not is_in_toc_zone(text, m.start())
        and not has_nearby_item_after(text, m.start(), radius=400)
    ]

    # For 7A only, stop here (skip cross-ref/“relating to” screens — too aggressive for short 7A)
    if label == "Item 7A":
        return cands

    # For 1/1A/7, keep stricter screens
    cands = [m for m in cands if not is_cross_reference_use(text, m.start(), lookahead=220)]
    cands = [m for m in cands if not RELATING_PAT.search(text[m.start(): m.start() + 100])]
    return cands

def compute_span_with_ender(text: str, start_idx: int, ender_pat: Optional[re.Pattern]) -> Tuple[int, int]:
    if ender_pat:
        m = ender_pat.search(text, pos=start_idx + 1)
        if m:
            return (start_idx, m.start())
    next_any = [m for m in ANY_ITEM_PATTERN.finditer(text) if m.start() > start_idx]
    end_idx = next_any[0].start() if next_any else len(text)
    return (start_idx, end_idx)

def find_alt_7a_starts(text: str) -> List[re.Match]:
    starts: List[re.Match] = []
    for pat in ALT_7A_TITLES_COMPILED:
        starts.extend(list(pat.finditer(text)))
    starts = [
        m for m in starts
        if not is_in_toc_zone(text, m.start())
        and not has_nearby_item_after(text, m.start(), radius=400)
    ]
    return starts

def _has_anchor_in_window(text: str, start: int, end: int, patterns: List[str], limit_chars: int = 5000) -> bool:
    win_end = min(end, start + limit_chars, len(text))
    window = text[start:win_end]
    for pat in patterns:
        if re.search(pat, window):
            return True
    return False

def best_span_for(label: str, text: str) -> Optional[Tuple[int, int]]:
    label_regex = HEADER_PATTERNS[label]
    ender_pat = SPECIFIC_ENDERS_COMPILED.get(label)

    # Strict pass
    starts = find_candidate_starts(text, label, label_regex)

    # Relaxed pass
    if not starts:
        lab_pat = re.compile(label_regex)
        if label == "Item 1":
            # Only exclude TOC; allow near-item proximity in case of dense headers
            starts = [m for m in lab_pat.finditer(text) if not is_in_toc_zone(text, m.start())]
        elif label == "Item 7A":
            starts = [
                m for m in lab_pat.finditer(text)
                if not is_in_toc_zone(text, m.start())
                and not has_nearby_item_after(text, m.start(), radius=400)
            ]

    # Final 7A fallback: spelled-out headings
    if (not starts) and (label == "Item 7A"):
        starts = find_alt_7a_starts(text)
        if not starts:
            return None

    if not starts:
        return None

    # Anchors per label
    anchors = ANCHORS_1 if label == "Item 1" else (
        ANCHORS_7 if label == "Item 7" else (
            ANCHORS_7A if label == "Item 7A" else ANCHORS_1A
        )
    )

    candidates: List[Tuple[Tuple[int, int], int]] = []
    for s in starts:
        s_idx = s.start()
        span = compute_span_with_ender(text, s_idx, ender_pat)
        cscore = content_score(text, span[0], span[1], anchors, label)
        candidates.append((span, cscore))

    candidates.sort(key=lambda x: (x[1], x[0][1] - x[0][0]), reverse=True)
    best_span, _ = candidates[0]
    best_len = best_span[1] - best_span[0]
    min_chars = MIN_CHARS.get(label, 0)

    if best_len >= min_chars or len(candidates) == 1:
        return best_span

    # If below threshold:
    if label == "Item 7":
        anchored = []
        for span, sc in candidates:
            if _has_anchor_in_window(text, span[0], span[1], ANCHORS_7, limit_chars=5000):
                anchored.append((span, sc))
        if anchored:
            anchored.sort(key=lambda x: (x[0][1] - x[0][0], x[1]), reverse=True)
            return anchored[0][0]
        candidates.sort(key=lambda x: (x[0][1] - x[0][0], x[1]), reverse=True)
        return candidates[0][0]

    if label == "Item 1":
        # Prefer candidates that show "Business" early
        biz = []
        for span, sc in candidates:
            s, e = span
            if re.search(r"(?i)\bBusiness\b", text[s: min(e, s + 5000)]):
                biz.append((span, sc))
        if biz:
            biz.sort(key=lambda x: (x[0][1] - x[0][0], x[1]), reverse=True)
            return biz[0][0]
        candidates.sort(key=lambda x: (x[0][1] - x[0][0], x[1]), reverse=True)
        return candidates[0][0]

    last_start = starts[-1].start()
    last_span = compute_span_with_ender(text, last_start, ender_pat)
    if (last_span[1] - last_span[0]) > best_len:
        return last_span

    candidates.sort(key=lambda x: (x[0][1] - x[0][0], x[1]), reverse=True)
    return candidates[0][0]

def extract_sections(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for label in ["Item 1", "Item 1A", "Item 7", "Item 7A"]:
        span = best_span_for(label, text)
        if not span:
            continue
        s, e = span
        out[label] = text[s:e].strip()
    return out

# ------------------ Orchestration ------------------
def to_rows(ticker: str, year: int, sections: Dict[str, str]) -> List[Dict]:
    rows = []
    for sec in ["Item 1", "Item 1A", "Item 7", "Item 7A"]:
        txt = sections.get(sec, "")
        if txt:
            rows.append({"ticker": ticker, "year": int(year), "section": sec, "text": txt})
    return rows

def main():
    ap = argparse.ArgumentParser(
        description="V2: Parse SEC 10-K filings to Parquet (Items 1, 1A, 7, 7A) with stronger selection and anti-TOC heuristics."
    )
    ap.add_argument("--raw-base", default="data/raw")
    ap.add_argument("--out-base", default="data/bronze_v2")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", default="logs_v2/parse_report_v2.csv")
    args = ap.parse_args()

    raw_base = Path(args.raw_base)
    out_base = Path(args.out_base)
    report_rows = []

    if not raw_base.exists():
        print(f"Raw base not found: {raw_base}", file=sys.stderr)
        sys.exit(1)

    tickers = args.tickers or [p.name for p in raw_base.iterdir() if p.is_dir()]
    for tkr in sorted(tickers):
        t_dir = raw_base / tkr
        if not t_dir.exists():
            continue
        years = args.years or [int(p.name) for p in t_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        for yr in sorted(years):
            in_dir = t_dir / str(yr)
            try:
                doc_path = auto_pick_html(in_dir)
            except FileNotFoundError as e:
                report_rows.append({
                    "ticker": tkr, "year": yr, "status": "NO_HTML", "details": str(e),
                    "source_doc": None,
                    "item_1": 0, "item_1A": 0, "item_7": 0, "item_7A": 0,
                    "chars_1": 0, "chars_1A": 0, "chars_7": 0, "chars_7A": 0,
                    "chars_total": 0, "short_1": None, "short_1A": None, "short_7": None, "short_7A": None
                })
                continue

            out_dir = out_base / tkr / str(yr)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "sections.parquet"

            if out_path.exists() and not args.overwrite:
                report_rows.append({
                    "ticker": tkr, "year": yr, "status": "SKIP_EXISTS",
                    "details": str(out_path), "source_doc": doc_path.name,
                    "item_1": None, "item_1A": None, "item_7": None, "item_7A": None,
                    "chars_1": None, "chars_1A": None, "chars_7": None, "chars_7A": None,
                    "chars_total": None, "short_1": None, "short_1A": None, "short_7": None, "short_7A": None
                })
                continue

            try:
                text = read_text_from_doc(doc_path)
                sections = extract_sections(text)
                rows = to_rows(tkr, yr, sections)

                got = {r["section"]: len(r["text"]) for r in rows}
                chars_total = sum(got.values()) if got else 0

                if rows:
                    df = pd.DataFrame(rows, columns=["ticker","year","section","text"])
                    df.to_parquet(out_path, index=False)

                    report_rows.append({
                        "ticker": tkr,
                        "year": yr,
                        "status": "OK",
                        "details": f"Wrote {len(rows)} sections to {out_path}",
                        "source_doc": doc_path.name,
                        "item_1": int("Item 1" in got),
                        "item_1A": int("Item 1A" in got),
                        "item_7": int("Item 7" in got),
                        "item_7A": int("Item 7A" in got),
                        "chars_1": got.get("Item 1", 0),
                        "chars_1A": got.get("Item 1A", 0),
                        "chars_7": got.get("Item 7", 0),
                        "chars_7A": got.get("Item 7A", 0),
                        "chars_total": chars_total,
                        "short_1": int(got.get("Item 1", 0) < MIN_CHARS["Item 1"]) if "Item 1" in got else None,
                        "short_1A": int(got.get("Item 1A", 0) < MIN_CHARS["Item 1A"]) if "Item 1A" in got else None,
                        "short_7": int(got.get("Item 7", 0) < MIN_CHARS["Item 7"]) if "Item 7" in got else None,
                        "short_7A": int(got.get("Item 7A", 0) < MIN_CHARS["Item 7A"]) if "Item 7A" in got else None,
                    })
                else:
                    report_rows.append({
                        "ticker": tkr, "year": yr, "status": "NO_SECTIONS",
                        "details": str(doc_path), "source_doc": doc_path.name,
                        "item_1": 0, "item_1A": 0, "item_7": 0, "item_7A": 0,
                        "chars_1": 0, "chars_1A": 0, "chars_7": 0, "chars_7A": 0,
                        "chars_total": 0, "short_1": None, "short_1A": None, "short_7": None, "short_7A": None
                    })

            except Exception as ex:
                report_rows.append({
                    "ticker": tkr, "year": yr, "status": "ERROR",
                    "details": f"{doc_path}: {ex}", "source_doc": doc_path.name,
                    "item_1": 0, "item_1A": 0, "item_7": 0, "item_7A": 0,
                    "chars_1": 0, "chars_1A": 0, "chars_7": 0, "chars_7A": 0,
                    "chars_total": 0, "short_1": None, "short_1A": None, "short_7": None, "short_7A": None
                })

    rep_path = Path(args.report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(rep_path, index=False)
    print(f"Done. Report → {rep_path}")
    print("Tip: focus rows where (item_1==0) or (short_7==1).")

if __name__ == "__main__":
    main()
