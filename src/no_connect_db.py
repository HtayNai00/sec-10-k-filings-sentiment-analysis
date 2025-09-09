# src/duckdb_no_connect.py
from pathlib import Path
import duckdb as d
import pandas as pd

P = Path("data/gold/finbert_section_scores.parquet")
print("cwd:", Path.cwd())
print("parquet exists:", P.exists(), "->", P.resolve())
if not P.exists():
    raise SystemExit("gold parquet not found")

def show(title, query):
    print("\n---", title, "---", flush=True)
    try:
        df = d.sql(query).df()  # materialize to pandas
        # limit width so it always prints
        with pd.option_context("display.max_rows", 20, "display.max_columns", 20, "display.width", 120):
            print(df.to_string(index=False), flush=True)
    except Exception as e:
        print("ERROR:", e, flush=True)

show("sanity", "select 1 as ok")

show("row count",
     f"select count(*) as n from read_parquet('{P.as_posix()}')")

show("peek 10",
     f"""
     select ticker, fiscal_year, item, n_sentences, pos_mean, neu_mean, neg_mean, polarity, label
     from read_parquet('{P.as_posix()}')
     limit 10
     """)

show("label distribution",
     f"""
     select item, label, count(*) as n
     from read_parquet('{P.as_posix()}')
     group by 1,2
     order by item, n desc
     """)
