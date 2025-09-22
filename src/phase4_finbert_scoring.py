#!/usr/bin/env python3
"""
Phase 4 — FinBERT scoring for SEC 10-K sections with optional entity collapsing

Inputs (silver):  Parquet files under data/silver/silver_1A7_v2/ (or a single parquet/glob)
Expected columns (rename via --col-* if yours differ):
  - ticker (str)
  - fiscal_year (int)      # e.g., 2020
  - filing_date (str)      # 'YYYY-MM-DD' (optional; synthesized if missing)
  - accession (str)        # (optional; synthesized if missing)
  - item (str)             # one of {'1A','7','1','7A'} — we will typically focus on 1A & 7
  - text (str)             # section text

Outputs (gold):
  - data/gold/finbert_section_scores.parquet
      One row per (ticker, fiscal_year, item, accession)
  - data/gold/finbert_sentence_scores.parquet  (optional --emit-sentences)
      One row per sentence with FinBERT probabilities & (optional) weights

Usage examples:
  python phase4_finbert_scoring.py \
    --silver-path data/silver/silver_1A7_v2 \
    --outdir data/gold \
    --batch-size 32 \
    --collapse-entities \
    --brand-alias-file configs/brand_aliases.json \
    --brand-heuristic-min 10 \
    --brand-density-threshold 0.25 \
    --temperature 0.85 \
    --resume \
    --emit-sentences

Notes:
  • Idempotent + resumable: pass --resume to skip already-scored keys.
  • Batches sentences and truncates >512 tokens automatically.
  • Choose model via --model (default ProsusAI/finbert).
  • Entity collapsing happens *before* sentence splitting; it is per-paragraph and preserves meaning.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# -------- sentence splitter (NLTK) --------
import nltk
from nltk.tokenize import sent_tokenize


# =========================
# NLTK setup
# =========================
def ensure_nltk():
    """Ensure NLTK tokenizers are available across NLTK versions."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


# =========================
# IO: read silver
# =========================
def read_silver(silver_path: str, col_map: Dict[str, str]) -> pd.DataFrame:
    p = Path(silver_path)
    files: List[Path] = []
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path().glob(silver_path))  # glob pattern
    if not files:
        raise FileNotFoundError(f"No parquet found at {silver_path}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Normalize/rename columns per map
    df = df.rename(columns={col_map[k]: k for k in col_map})

    # Flexible aliases to match your silver schema
    if 'fiscal_year' not in df.columns and 'year' in df.columns:
        df['fiscal_year'] = df['year']
    if 'item' not in df.columns and 'section' in df.columns:
        df['item'] = df['section']

    # filing_date/accession optional; synthesize if missing
    if 'filing_date' not in df.columns:
        df['filing_date'] = pd.to_datetime(df['fiscal_year'].astype(int).astype(str) + '-01-01').dt.strftime('%Y-%m-%d')
    if 'accession' not in df.columns:
        df['accession'] = df['ticker'].astype(str) + '-' + df['fiscal_year'].astype(int).astype(str)

    required = ["ticker", "fiscal_year", "item", "text"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    # Clean/normalize
    df = df.dropna(subset=["text"]).copy()
    df["item"] = df["item"].astype(str).str.upper().str.replace("ITEM ", "", regex=False)
    df["item"] = df["item"].str.replace("ITEM", "", regex=False).str.strip()
    df["item"] = df["item"].replace({"1A.": "1A", "7.": "7", "1.": "1", "7A.": "7A"})

    # Keep expected items if present (you can later filter to only 1A & 7 via --items)
    df = df[df["item"].isin(["1A", "7", "1", "7A"])]
    return df


# =========================
# HF pipeline
# =========================
def build_pipeline(model_name: str, use_gpu: int) -> TextClassificationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = -1
    if use_gpu and torch.cuda.is_available():
        device = 0
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
        truncation=True,
        max_length=512,
        padding=True,
    )
    return pipe


LABELS = ["positive", "neutral", "negative"]


def scores_to_vector(all_scores: List[Dict[str, float]]) -> Tuple[float, float, float]:
    # map list of dicts -> [pos, neu, neg]
    by_label = {d["label"].lower(): d["score"] for d in all_scores}
    return (
        float(by_label.get("positive", 0.0)),
        float(by_label.get("neutral", 0.0)),
        float(by_label.get("negative", 0.0)),
    )


def majority_label(pos_mean: float, neu_mean: float, neg_mean: float) -> str:
    arr = np.array([pos_mean, neu_mean, neg_mean])
    idx = int(arr.argmax())
    return LABELS[idx]


# =========================
# Entity collapsing
# =========================
CAP_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9&\-\.’']{1,}\b")  # simplistic capitalized token
WORD = re.compile(r"[A-Za-z]+")

WHITELIST_NE = {
    "United States", "U.S.", "US", "USA", "SEC", "GAAP", "IFRS", "FASB", "OECD",
    "NYSE", "NASDAQ", "S&P", "Moody's", "Fitch", "COVID-19", "LIBOR", "SOFR",
}

def load_aliases_map(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Load alias map from JSON or CSV.
    JSON: {"AAPL": ["Apple", "Apple Inc.", "iPhone", "MacBook", ...], ...}
    CSV headers: ticker,alias
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Alias file not found: {path}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if not {"ticker", "alias"}.issubset(df.columns):
            raise ValueError("CSV alias file must have columns: ticker, alias")
        out: Dict[str, List[str]] = {}
        for t, a in df[["ticker", "alias"]].itertuples(index=False):
            out.setdefault(str(t), []).append(str(a))
        return out
    else:
        raise ValueError("Alias file must be .json or .csv")


def discover_capitalized_aliases(text: str, min_count: int) -> List[str]:
    """
    Discover repeated capitalized tokens/phrases as weak aliases.
    Very lightweight heuristic; ignores whitelisted finance/legal terms.
    """
    counts: Dict[str, int] = {}
    # Consider per-line to catch multi-word names
    for line in text.splitlines():
        # naive multi-word capture: join consecutive CapTokens
        tokens = CAP_TOKEN.findall(line)
        if not tokens:
            continue
        phrase = []
        for tok in tokens:
            phrase.append(tok)
        # count singles and simple multi-word collocations (up to length 3)
        for i, tok in enumerate(tokens):
            if tok in WHITELIST_NE:
                continue
            counts[tok] = counts.get(tok, 0) + 1
            if i + 1 < len(tokens):
                two = f"{tokens[i]} {tokens[i+1]}"
                if two not in WHITELIST_NE:
                    counts[two] = counts.get(two, 0) + 1
            if i + 2 < len(tokens):
                three = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                if three not in WHITELIST_NE:
                    counts[three] = counts.get(three, 0) + 1

    aliases = [k for k, v in counts.items() if v >= min_count and k not in WHITELIST_NE]
    # Sort by length desc so longer phrases collapse first
    aliases.sort(key=lambda s: (-len(s), s))
    return aliases


def _pluralize_stub(stub: str, original: str) -> str:
    # Preserve simple plural/possessive endings when replacing with <BRAND>
    if original.endswith("'s") or original.endswith("’s"):
        return stub + "'s"
    if original.endswith("s") and not original.endswith("'s"):
        return stub + "s"
    return stub


def collapse_paragraph(paragraph: str, aliases: List[str], cap_per_para: int = 3) -> str:
    """
    Replace repeated mentions of any alias in the SAME paragraph with <BRAND>.
    Keep up to cap_per_para mentions verbatim (first occurrences).
    Preserve possessives/plurals when collapsing.
    """
    if not paragraph.strip() or not aliases:
        return paragraph

    # Build regex for aliases (word boundary-ish), escape special characters
    escaped = [re.escape(a) for a in aliases if a.strip()]
    if not escaped:
        return paragraph

    # Sort by length desc to prefer longer matches first
    escaped.sort(key=lambda s: (-len(s), s))
    pattern = re.compile(r"(" + "|".join(escaped) + r")(?=[^\w]|$)", flags=re.U)

    keep_budget = cap_per_para
    out = []
    last = 0
    seen_positions = 0

    for m in pattern.finditer(paragraph):
        out.append(paragraph[last:m.start()])
        original = m.group(0)
        if keep_budget > 0:
            # keep this exact surface form
            out.append(original)
            keep_budget -= 1
        else:
            # collapse to placeholder, preserving simple endings
            out.append(_pluralize_stub("<BRAND>", original))
        last = m.end()
        seen_positions += 1

    out.append(paragraph[last:])
    return "".join(out)


def collapse_entities_in_text(
    text: str,
    ticker: str,
    item: str,
    aliases_map: Dict[str, List[str]],
    heuristic_min: int = 10,
    cap_per_para: int = 3,
) -> str:
    """
    Collapses brand/product mentions within each paragraph to reduce neutral inflation.
    Strategy:
      • seed aliases from provided map (per ticker), plus
      • heuristic discovery of frequent capitalized tokens/phrases.
    Then, per paragraph: keep first 'cap_per_para' mentions, collapse the rest to <BRAND>.
    """
    # Seed aliases
    aliases = [a for a in aliases_map.get(ticker, []) if isinstance(a, str)]
    # Heuristic discovery (item scoped)
    discovered = discover_capitalized_aliases(text, min_count=heuristic_min)
    # Merge, keep unique, longer phrases first
    merged = list(dict.fromkeys([*aliases, *discovered]))
    merged.sort(key=lambda s: (-len(s), s))

    if not merged:
        return text

    # Paragraph-wise collapsing
    paras = re.split(r"(\n\s*\n+)", text)  # keep delimiters so structure is preserved
    for i in range(0, len(paras), 2):
        paras[i] = collapse_paragraph(paras[i], merged, cap_per_para=cap_per_para)
    return "".join(paras)


# =========================
# Sentence pre-filtering & weighting
# =========================
def pre_filter_sentences(sents: List[str], min_len: int = 25) -> List[str]:
    """Merge tiny fragments; drop micro-sentences and all-caps stubs."""
    out = []
    buf = ""
    for s in sents:
        t = s.strip()
        if not t:
            continue
        if len(t) < min_len:
            buf = (buf + " " + t).strip()
            continue
        if buf:
            t = (buf + " " + t).strip()
            buf = ""
        if len(t) < min_len and t.isupper():
            continue
        out.append(t)
    if buf:
        out.append(buf)
    return out


def brand_density_weight(sentence: str, threshold: float = 0.25) -> float:
    """Down-weight sentences dominated by <BRAND> tokens."""
    if "<BRAND>" not in sentence:
        return 1.0
    tokens = sentence.split()
    if not tokens:
        return 1.0
    d = sum(1 for t in tokens if "<BRAND>" in t) / len(tokens)
    if d <= threshold:
        return 1.0
    # Linear falloff: at d=1.0 weight ~0.2
    return float(max(0.2, 1.0 - (d - threshold) * 1.2))


def retune_neutral(pos_mean: float, neu_mean: float, neg_mean: float, temperature: Optional[float]) -> Tuple[float, float, float]:
    """Optional mild temperature to reduce neutral wash-out."""
    if not temperature or temperature <= 0:
        return pos_mean, neu_mean, neg_mean
    p = np.array([pos_mean, neu_mean, neg_mean], dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p) - np.log(1 - p)
    logit = logit / float(temperature)
    ex = np.exp(logit - logit.max())
    q = ex / ex.sum()
    return float(q[0]), float(q[1]), float(q[2])


# =========================
# Scoring
# =========================
def score_sentences(pipe: TextClassificationPipeline, sentences: List[str], batch_size: int = 32):
    probs = []  # list of (pos, neu, neg)
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        out = pipe(batch)
        for r in out:
            pos, neu, neg = scores_to_vector(r)
            probs.append((pos, neu, neg))
            results.append(r)
    return np.array(probs) if probs else np.empty((0, 3))


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Phase 4 — FinBERT scoring (with optional entity collapsing)")
    parser.add_argument("--silver-path", type=str, default="data/silver/silver_1A7_v2", help="Parquet file, folder, or glob for silver input")
    parser.add_argument("--outdir", type=str, default="data/gold", help="Output directory for gold parquet files")
    parser.add_argument("--model", type=str, default="ProsusAI/finbert", help="HF model name (e.g., ProsusAI/finbert or yiyanghkust/finbert-tone)")
    parser.add_argument("--batch-size", type=int, default=32, help="Sentence batch size for inference")
    parser.add_argument("--use-gpu", type=int, default=1, help="1 to use CUDA if available, else CPU")
    parser.add_argument("--emit-sentences", action="store_true", help="Also write sentence-level parquet")
    parser.add_argument("--resume", action="store_true", help="Skip keys already present in section parquet")

    # Column name overrides for your silver schema
    parser.add_argument("--col-ticker", type=str, default="ticker")
    parser.add_argument("--col-year", type=str, default="fiscal_year")
    parser.add_argument("--col-date", type=str, default="filing_date")
    parser.add_argument("--col-accession", type=str, default="accession")
    parser.add_argument("--col-item", type=str, default="item")
    parser.add_argument("--col-text", type=str, default="text")

    # Focus items (default to only 1A & 7 to match your plan)
    parser.add_argument("--items", nargs="*", default=["1A", "7"], help="Which items to score (default: 1A 7). Examples: 1A 7 1 7A")

    # Entity collapsing & weighting
    parser.add_argument("--collapse-entities", action="store_true", help="Enable brand/product entity collapsing with <BRAND>")
    parser.add_argument("--brand-alias-file", type=str, default=None, help="JSON or CSV with columns {ticker,alias} or JSON map")
    parser.add_argument("--brand-heuristic-min", type=int, default=10, help="Min repeats to treat a capitalized token/phrase as an alias")
    parser.add_argument("--brand-cap-per-paragraph", type=int, default=3, help="Mentions kept verbatim per paragraph before collapsing")
    parser.add_argument("--brand-density-threshold", type=float, default=0.25, help="Down-weight sentences with many <BRAND> tokens above this ratio")

    # Sentence hygiene & neutral re-tune
    parser.add_argument("--min-sent-len", type=int, default=25, help="Drop/merge sentences shorter than this length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Neutral de-biasing temperature (0 disables; try 0.85)")

    args = parser.parse_args()

    ensure_nltk()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    section_out = outdir / "finbert_section_scores.parquet"
    sentence_out = outdir / "finbert_sentence_scores.parquet"

    col_map = {
        "ticker": args.col_ticker,
        "fiscal_year": args.col_year,
        "filing_date": args.col_date,
        "accession": args.col_accession,
        "item": args.col_item,
        "text": args.col_text,
    }

    print("[Phase4] Loading silver…")
    df = read_silver(args.silver_path, col_map)

    # Filter to items of interest
    focus_items = set([s.upper() for s in args.items])
    df = df[df["item"].isin(focus_items)].copy()
    if df.empty:
        raise SystemExit(f"No rows left after filtering to items={sorted(focus_items)}")

    # Prepare resume set
    done_keys = set()
    if args.resume and section_out.exists():
        try:
            existing = pd.read_parquet(section_out)
            existing = existing[existing["item"].isin(focus_items)]
            done_keys = set(zip(existing["ticker"], existing["fiscal_year"], existing["accession"], existing["item"]))
            print(f"[Phase4] Resume enabled — {len(done_keys)} keys already scored for selected items.")
        except Exception as e:
            print(f"[Phase4] Could not read existing section parquet for resume: {e}")

    # Build model pipeline
    print(f"[Phase4] Loading model: {args.model}")
    pipe = build_pipeline(args.model, args.use_gpu)

    # Load alias map if provided
    aliases_map = load_aliases_map(args.brand_alias_file) if args.collapse_entities else {}

    # Accumulators
    section_rows = []
    sentence_rows = [] if args.emit_sentences else None

    # Sort for determinism
    df = df.sort_values(["ticker", "fiscal_year", "filing_date", "item"]).reset_index(drop=True)

    pbar = tqdm(total=len(df), desc="Scoring sections")
    for _, row in df.iterrows():
        key = (row["ticker"], int(row["fiscal_year"]), row["accession"], row["item"])
        if args.resume and key in done_keys:
            pbar.update(1)
            continue

        text = str(row["text"]).strip()
        if not text:
            pbar.update(1)
            continue

        # ---- Entity collapsing (optional) ----
        if args.collapse_entities:
            text = collapse_entities_in_text(
                text=text,
                ticker=str(row["ticker"]),
                item=str(row["item"]),
                aliases_map=aliases_map,
                heuristic_min=int(args.brand_heuristic_min),
                cap_per_para=int(args.brand_cap_per_paragraph),
            )

        # Split to sentences (fast + simple) + pre-filter
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        sentences = pre_filter_sentences(sentences, min_len=args.min_sent_len)
        if not sentences:
            pbar.update(1)
            continue

        # Per-sentence weights based on <BRAND> density (optional)
        weights = np.array(
            [brand_density_weight(s, threshold=args.brand_density_threshold) if args.collapse_entities else 1.0 for s in sentences],
            dtype=float,
        )

        probs = score_sentences(pipe, sentences, batch_size=args.batch_size)  # (n, 3)
        if probs.size == 0:
            pbar.update(1)
            continue

        # Weighted means (avoid all-zero weights)
        if weights.sum() <= 1e-9:
            weights = np.ones_like(weights)
        w = weights / weights.sum()
        pos_mean = float((probs[:, 0] * w).sum())
        neu_mean = float((probs[:, 1] * w).sum())
        neg_mean = float((probs[:, 2] * w).sum())

        # Optional neutral de-biasing (mild)
        pos_mean, neu_mean, neg_mean = retune_neutral(pos_mean, neu_mean, neg_mean, temperature=args.temperature)

        label = majority_label(pos_mean, neu_mean, neg_mean)
        polarity = pos_mean - neg_mean

        section_rows.append({
            "ticker": row["ticker"],
            "fiscal_year": int(row["fiscal_year"]),
            "filing_date": row["filing_date"],
            "accession": row["accession"],
            "item": row["item"],
            "n_sentences": int(len(sentences)),
            "pos_mean": pos_mean,
            "neu_mean": neu_mean,
            "neg_mean": neg_mean,
            "polarity": polarity,
            "label": label,
            "model": args.model,
            "collapse_entities": bool(args.collapse_entities),
            "brand_density_threshold": float(args.brand_density_threshold) if args.collapse_entities else None,
            "temperature": float(args.temperature) if args.temperature else None,
        })

        if sentence_rows is not None:
            for i, (s, (p, u, n), wt) in enumerate(zip(sentences, probs, weights)):
                sentence_rows.append({
                    "ticker": row["ticker"],
                    "fiscal_year": int(row["fiscal_year"]),
                    "filing_date": row["filing_date"],
                    "accession": row["accession"],
                    "item": row["item"],
                    "sent_idx": i,
                    "sentence": s,
                    "pos": float(p),
                    "neu": float(u),
                    "neg": float(n),
                    "weight": float(wt),
                    "label": LABELS[int(np.argmax([p, u, n]))],
                    "model": args.model,
                    "collapse_entities": bool(args.collapse_entities),
                })

            # Optional periodic flush (kept as in your codebase)
            if len(sentence_rows) >= 50_000:
                df_sent = pd.DataFrame(sentence_rows)
                if sentence_out.exists():
                    # NOTE: Pandas/pyarrow doesn't truly support append in to_parquet across engines.
                    # We keep your pattern; if your environment errors, switch to concatenating and rewriting.
                    df_sent.to_parquet(sentence_out, index=False, compression="zstd", append=True)
                else:
                    df_sent.to_parquet(sentence_out, index=False, compression="zstd")
                sentence_rows.clear()

        # Periodic flush for sections
        if len(section_rows) >= 2_000:
            df_sec = pd.DataFrame(section_rows)
            if section_out.exists():
                df_sec.to_parquet(section_out, index=False, compression="zstd", append=True)
            else:
                df_sec.to_parquet(section_out, index=False, compression="zstd")
            section_rows.clear()

        pbar.update(1)

    pbar.close()

    # Final flush
    if section_rows:
        df_sec = pd.DataFrame(section_rows)
        if section_out.exists():
            df_sec.to_parquet(section_out, index=False, compression="zstd", append=True)
        else:
            df_sec.to_parquet(section_out, index=False, compression="zstd")

    if sentence_rows:
        df_sent = pd.DataFrame(sentence_rows)
        if sentence_out.exists():
            df_sent.to_parquet(sentence_out, index=False, compression="zstd", append=True)
        else:
            df_sent.to_parquet(sentence_out, index=False, compression="zstd")

    print("[Phase4] Done. Wrote:")
    print(f"  • {section_out}")
    if args.emit_sentences:
        print(f"  • {sentence_out}")


if __name__ == "__main__":
    main()
