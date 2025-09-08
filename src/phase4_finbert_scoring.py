#!/usr/bin/env python3
"""
Phase 4 — FinBERT scoring for SEC 10‑K sections

Inputs (silver):  Parquet files under data/silver/silver_1A7_v2/
Expected columns (rename in --col-* flags if yours differ):
  - ticker (str)
  - fiscal_year (int)      # e.g., 2020
  - filing_date (str)      # 'YYYY-MM-DD'
  - accession (str)        # or accession_number
  - item (str)             # one of {'1A','7','1','7A'}
  - text (str)             # section text

Outputs (gold):
  - data/gold/finbert_section_scores.parquet
      one row per (ticker, fiscal_year, item, accession)
  - data/gold/finbert_sentence_scores.parquet  (optional --emit-sentences)
      one row per sentence with FinBERT probabilities

Usage examples:
  python phase4_finbert_scoring.py \
    --silver-path data/silver/silver_1A7_v2 \
    --outdir data/gold \
    --batch-size 32 --emit-sentences --resume

  # CPU only
  python phase4_finbert_scoring.py --use-gpu 0

Notes:
  • Idempotent + resumable: pass --resume to skip already-scored keys.
  • Batches sentences and truncates >512 tokens automatically.
  • Choose model via --model (default ProsusAI/finbert).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# -------- sentence splitter (NLTK) --------
import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk():
    """Ensure NLTK tokenizers are available across NLTK versions.
    Newer NLTK releases split data under 'punkt_tab'. We attempt both.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Some installations require an extra 'punkt_tab' resource.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            # If punkt_tab isn't available in this NLTK build, ignore —
            # sent_tokenize will still work with 'punkt' on many versions.
            pass


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

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Normalize/rename columns per map
    df = df.rename(columns={col_map[k]: k for k in col_map})

    # Flexible aliases to match your silver schema
    if 'fiscal_year' not in df.columns and 'year' in df.columns:
        df['fiscal_year'] = df['year']
    if 'item' not in df.columns and 'section' in df.columns:
        df['item'] = df['section']

    # Make filing_date/accession optional; synthesize if missing
    if 'filing_date' not in df.columns:
        # fallback placeholder: Jan 1 of fiscal_year (refine in Phase 5 when joining metadata)
        df['filing_date'] = pd.to_datetime(df['fiscal_year'].astype(int).astype(str) + '-01-01').dt.strftime('%Y-%m-%d')
    if 'accession' not in df.columns:
        df['accession'] = df['ticker'].astype(str) + '-' + df['fiscal_year'].astype(int).astype(str)

    required = ["ticker", "fiscal_year", "item", "text"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Available: {list(df.columns)}")

    # Clean
    df = df.dropna(subset=["text"]).copy()
    df["item"] = df["item"].astype(str).str.upper().str.replace("ITEM ", "", regex=False)
    df["item"] = df["item"].str.replace("ITEM", "", regex=False).str.strip()
    df["item"] = df["item"].replace({"1A.": "1A", "7.": "7", "1.": "1", "7A.": "7A"})

    # Keep only expected items if present
    df = df[df["item"].isin(["1A", "7", "1", "7A"])]
    return df


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


LABEL_ORDER = ["positive", "neutral", "negative"]  # ProsusAI/finbert label order varies; we will map by name


def scores_to_vector(all_scores: List[Dict[str, float]]) -> Tuple[float, float, float]:
    # map list of dicts -> [pos, neu, neg]
    by_label = {d["label"].lower(): d["score"] for d in all_scores}
    return (
        float(by_label.get("positive", 0.0)),
        float(by_label.get("neutral", 0.0)),
        float(by_label.get("negative", 0.0)),
    )


def score_sentences(pipe: TextClassificationPipeline, sentences: List[str], batch_size: int = 32):
    probs = []  # list of (pos, neu, neg)
    # Run in batches to avoid OOM
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            pos, neu, neg = scores_to_vector(r)
            probs.append((pos, neu, neg))
    return np.array(probs) if probs else np.empty((0, 3))


def majority_label(pos_mean: float, neu_mean: float, neg_mean: float) -> str:
    arr = np.array([pos_mean, neu_mean, neg_mean])
    idx = int(arr.argmax())
    return ["positive", "neutral", "negative"][idx]


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — FinBERT scoring")
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

    # Prepare resume set
    done_keys = set()
    if args.resume and section_out.exists():
        try:
            existing = pd.read_parquet(section_out)
            done_keys = set(zip(existing["ticker"], existing["fiscal_year"], existing["accession"], existing["item"]))
            print(f"[Phase4] Resume enabled — {len(done_keys)} keys already scored.")
        except Exception as e:
            print(f"[Phase4] Could not read existing section parquet for resume: {e}")

    # Build model pipeline
    print(f"[Phase4] Loading model: {args.model}")
    pipe = build_pipeline(args.model, args.use_gpu)

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

        # Split to sentences (fast + simple)
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if not sentences:
            pbar.update(1)
            continue

        probs = score_sentences(pipe, sentences, batch_size=args.batch_size)  # (n,3)
        pos_mean = float(probs[:, 0].mean()) if len(probs) else 0.0
        neu_mean = float(probs[:, 1].mean()) if len(probs) else 0.0
        neg_mean = float(probs[:, 2].mean()) if len(probs) else 0.0
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
        })

        if sentence_rows is not None:
            # Store per sentence rows (chunk flush every ~50k sentences to control memory)
            for i, (s, (p, u, n)) in enumerate(zip(sentences, probs)):
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
                    "label": ["positive", "neutral", "negative"][int(np.argmax([p, u, n]))],
                    "model": args.model,
                })
            if len(sentence_rows) >= 50_000:
                df_sent = pd.DataFrame(sentence_rows)
                if sentence_out.exists():
                    df_sent.to_parquet(sentence_out, index=False, compression="zstd", append=True)
                else:
                    df_sent.to_parquet(sentence_out, index=False, compression="zstd")
                sentence_rows.clear()

        # Periodically flush section rows to parquet to avoid loss
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
