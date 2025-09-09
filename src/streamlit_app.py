import pandas as pd, streamlit as st
df = pd.read_parquet("data/gold/finbert_section_scores.parquet")
t = st.selectbox("Ticker", sorted(df.ticker.unique()))
y = st.selectbox("Year", sorted(df.loc[df.ticker==t,"fiscal_year"].unique()))
i = st.selectbox("Item", ["1A","7"])
row = df[(df.ticker==t)&(df.fiscal_year==y)&(df.item==i)].iloc[0]
st.metric("Polarity", f"{row.polarity:.3f}")
st.write({"pos":row.pos_mean, "neu":row.neu_mean, "neg":row.neg_mean, "label":row.label})
