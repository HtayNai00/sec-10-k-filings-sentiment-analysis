# src/duckdb_smoke.py
from pathlib import Path
import duckdb as d

print(">>> start")
print(">>> cwd:", Path.cwd())
p = Path("data/gold/finbert_section_scores.parquet")
print(">>> parquet exists?", p.exists(), "->", p.resolve())

print(">>> connecting (in-memory)")
con = d.connect()  # in-memory DB (no file on disk)
print(">>> connected")

# Create an in-memory view (zero-copy over parquet)
con.sql("""
CREATE OR REPLACE VIEW gold_filing_section_sentiment AS
SELECT * FROM read_parquet('data/gold/finbert_section_scores.parquet');
""")
print(">>> view created")

print(">>> row count")
con.sql("SELECT COUNT(*) AS n FROM gold_filing_section_sentiment").show()

print(">>> peek 10")
con.sql("""
SELECT ticker, fiscal_year, item, n_sentences, pos_mean, neu_mean, neg_mean, polarity, label
FROM gold_filing_section_sentiment
LIMIT 10
""").show()

print(">>> done")
