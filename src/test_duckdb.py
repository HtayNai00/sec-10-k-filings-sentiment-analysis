# src/test_duckdb.py
from pathlib import Path
import sys
import duckdb as d

SECTION_PARQUET = Path("data/gold/finbert_section_scores.parquet")
RETURNS_PARQUET = Path("data/gold/filing_returns.parquet")  # optional

# 0) Check files exist and show absolute paths
print("Working dir:", Path.cwd())
print("Section parquet:", SECTION_PARQUET.resolve())
if not SECTION_PARQUET.exists():
    print("❌ Not found. Run Phase 4 or fix the path.")
    sys.exit(1)

# 1) Connect (creates a small DuckDB file next to your gold)
con = d.connect("data/gold/sec_gold.duckdb")

# 2) Create views over parquet (instant; zero-copy)
con.sql(f"""
CREATE OR REPLACE VIEW gold_filing_section_sentiment AS
SELECT * FROM read_parquet('{SECTION_PARQUET.as_posix()}');
""")

if RETURNS_PARQUET.exists():
    con.sql(f"""
    CREATE OR REPLACE VIEW gold_filing_returns AS
    SELECT * FROM read_parquet('{RETURNS_PARQUET.as_posix()}');
    """)

# 3) Actually SHOW results
print("\n--- Count rows in section parquet ---")
con.sql("SELECT COUNT(*) AS n FROM gold_filing_section_sentiment").show()

print("\n--- Peek 10 rows ---")
con.sql("""
SELECT ticker, fiscal_year, item, n_sentences, pos_mean, neu_mean, neg_mean, polarity, label
FROM gold_filing_section_sentiment
LIMIT 10
""").show()

print("\n--- Label distribution by item ---")
con.sql("""
SELECT item, label, COUNT(*) AS n
FROM gold_filing_section_sentiment
GROUP BY 1,2
ORDER BY item, n DESC
""").show()

# 4) Optional: returns (only if Phase 5 exists)
if RETURNS_PARQUET.exists():
    print("\n--- Returns joined (first 10) ---")
    con.sql("""
    SELECT s.ticker, s.fiscal_year, s.item, s.polarity,
           r.filing_date, r.buy_date, r.ret_30d, r.ret_60d, r.ret_90d
    FROM gold_filing_section_sentiment s
    JOIN gold_filing_returns r
      ON s.ticker = r.ticker AND s.fiscal_year = r.fiscal_year
    WHERE s.item IN ('1A','7')
    LIMIT 10
    """).show()
else:
    print("\n(no returns yet — run Phase 5 to create data/gold/filing_returns.parquet)")
