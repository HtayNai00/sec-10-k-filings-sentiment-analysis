import pandas as pd

df = pd.read_parquet("data/gold_v2/finbert_section_scores.parquet")
print(df.shape)
print(df.head())
