# src/ingest_edgar.py
"""
Download SEC 10-K filings (not 10-K/A) for a list of tickers and a date range.
Reads tickers from tickers.txt (one per line) by default.
Saves the primary filing document to data/raw/<TICKER>/<YEAR>/.

Run from project root:
  python src/ingest_edgar.py --tickers-file tickers.txt
  # or test one
  python src/ingest_edgar.py --tickers AAPL --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations
import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import date
from dateutil.parser import parse as dateparse

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ----------------- CORRECT ENDPOINTS -----------------
SEC_FILES_URL = "https://www.sec.gov/files/company_tickers.json"      # ticker→CIK mapping
SEC_SUBMISSIONS_BASE = "https://data.sec.gov"                         # submissions JSON
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"         # filing documents
# -----------------------------------------------------

# ✅ Your real contact info (required by SEC fair access)
USER_AGENT = "HtayNaing Oo (Capstone Project; hnaing@student.ysu.edu)"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    # DO NOT set "Host" here; it breaks cross-host requests.
}

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_OUTDIR = "data/raw"
DEFAULT_SLEEP = 0.35  # seconds between HTTP requests


def read_tickers(tickers_arg: str | None, tickers_file: str | None) -> List[str]:
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    if tickers_file and Path(tickers_file).exists():
        return [line.strip().upper() for line in Path(tickers_file).read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")]
    raise SystemExit("No tickers provided. Use --tickers AAPL,MSFT or --tickers-file tickers.txt")


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.headers.update(HEADERS)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def fetch_ticker_to_cik(session: requests.Session) -> Dict[str, str]:
    r = session.get(SEC_FILES_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def load_company_submissions(session: requests.Session, cik_str: str) -> dict:
    url = f"{SEC_SUBMISSIONS_BASE}/submissions/CIK{cik_str}.json"
    print(f"Fetching submissions JSON: {url}")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def build_primary_doc_url(cik_nozero: str, accession_no: str, primary_doc: str) -> str:
    acc_no_nodash = accession_no.replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{cik_nozero}/{acc_no_nodash}/{primary_doc}"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def within_range(iso_date: str, start: date, end: date) -> bool:
    d = dateparse(iso_date).date()
    return start <= d <= end


def download_primary(session: requests.Session, ticker: str, cik: str,
                     row: Tuple[str, str, str, str], out_root: Path,
                     sleep_sec: float) -> bool:
    form, filing_date, accession, primary_doc = row
    year = dateparse(filing_date).year
    dest_dir = out_root / ticker / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = build_primary_doc_url(cik.lstrip("0"), accession, primary_doc)
    ext = os.path.splitext(primary_doc)[1] or ".html"
    fname = safe_name(f"{ticker}_{filing_date}_{accession}{ext}")
    dest_path = dest_dir / fname

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    try:
        resp = session.get(url, timeout=60)
        if resp.status_code == 200 and resp.content:
            dest_path.write_bytes(resp.content)
        else:
            (dest_dir / f"ERROR_{fname}.txt").write_text(f"HTTP {resp.status_code} for {url}\n")
            return False
    except Exception as e:
        (dest_dir / f"ERROR_{fname}.txt").write_text(f"{e}\nURL: {url}\n")
        return False
    finally:
        time.sleep(sleep_sec)

    return True


def main():
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings to data/raw/")
    parser.add_argument("--tickers", help="Comma-separated tickers, e.g., AAPL,MSFT,GOOGL")
    parser.add_argument("--tickers-file", default="tickers.txt", help="File with tickers (one per line)")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output root directory")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Delay between requests (sec)")
    args = parser.parse_args()

    tickers = read_tickers(args.tickers, args.tickers_file)
    start_d = dateparse(args.start).date()
    end_d = dateparse(args.end).date()
    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Tickers: {tickers}")
    print(f"Date range: {start_d} → {end_d}")

    session = make_session()
    print("Fetching ticker→CIK mapping…")
    t2c = fetch_ticker_to_cik(session)

    total_downloaded = 0
    for ticker in tickers:
        cik = t2c.get(ticker)
        if not cik:
            print(f"[WARN] No CIK for {ticker}, skipping.")
            continue

        # Load main submissions JSON
        data = load_company_submissions(session, cik)

        # Start with 'recent' arrays
        rec = data.get("filings", {}).get("recent", {})
        forms = list(rec.get("form", []))
        dates = list(rec.get("filingDate", []))
        accessions = list(rec.get("accessionNumber", []))
        primaries = list(rec.get("primaryDocument", []))

        # Merge in older pages listed under 'files'
        for f in data.get("filings", {}).get("files", []):
            # f["name"] typically like 'CIK0000320193-2020.json' or 'filings-2021.json'
            older_url = f'{SEC_SUBMISSIONS_BASE}/submissions/{f["name"]}'
            r = session.get(older_url, timeout=30)
            if not r.ok:
                continue
            older_recent = r.json().get("filings", {}).get("recent", {})
            forms.extend(older_recent.get("form", []))
            dates.extend(older_recent.get("filingDate", []))
            accessions.extend(older_recent.get("accessionNumber", []))
            primaries.extend(older_recent.get("primaryDocument", []))

        # Combine, keep only 10-K in date range
        rows = list(zip(forms, dates, accessions, primaries))
        rows = [r for r in rows if r[0] == "10-K"]
        rows = [r for r in rows if within_range(r[1], start_d, end_d)]

        if not rows:
            print(f"[INFO] No 10-Ks in range for {ticker}.")
            continue

        print(f"[{ticker}] {len(rows)} filing(s) to download.")
        for row in tqdm(rows, ncols=100):
            if download_primary(session, ticker, cik, row, out_root, args.sleep):
                total_downloaded += 1

    print(f"Done. Saved/verified {total_downloaded} file(s) under {out_root}")


if __name__ == "__main__":
    main()
