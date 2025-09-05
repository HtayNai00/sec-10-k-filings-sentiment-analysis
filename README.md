# 📊 SEC 10-K Filings Sentiment Analysis

**Capstone Project** | Natural Language Processing + Data Engineering + Finance  

This project builds an **end-to-end pipeline** for analyzing sentiment in **SEC 10-K filings** (2020–2024) for 48 major U.S. companies, and studying how disclosure sentiment relates to stock price performance.

---

## 📌 Project Overview
Public companies must file **Form 10-K** annually with the SEC. These reports include management discussion, risk factors, and financial data.  

Using **NLP (FinBERT)** and a structured **data engineering pipeline**, this project:

1. **Ingests 10-K filings** from SEC EDGAR.  
2. **Parses key sections**:  
   - Item 1A → *Risk Factors*  
   - Item 7 → *Management’s Discussion & Analysis*  
   - Item 7A → *Market Risk Disclosures*  
3. **Splits into sentences** and applies **FinBERT sentiment analysis**.  
4. **Merges with market data** (stock price returns: 1 week, 1 month).  
5. **Builds features** for analytics.  
6. **Logs experiments with MLflow** for reproducibility.  
7. **Visualizes results in a Streamlit dashboard**.  

---

## 🏗️ Repository Structure

sec10k_project/
├── src/ # Python scripts
│ ├── ingest_edgar.py # Download SEC 10-Ks
│ ├── parse_sections.py # Extract narrative sections
│ ├── sentiment_finbert.py # Apply FinBERT sentiment
│ ├── join_prices.py # Merge filings with stock returns
│ ├── build_features.py # Aggregate sentiment + returns
│ └── pipeline.py # Orchestrate full pipeline with MLflow
│
├── dashboard/
│ └── app.py # Streamlit dashboard app
│
├── docker/
│ └── Dockerfile # Containerized setup
│
├── data/ # (ignored) raw/silver/gold datasets
├── mlruns/ # (ignored) MLflow experiment tracking
├── requirements.txt # Python dependencies
├── tickers.txt # List of 48 company tickers
├── .gitignore
└── README.md


---

## ⚙️ Installation
### 1. Clone repo
```bash
git clone https://github.com/HtayNai00/sec-10-k-filings-sentiment-analysis.git
cd sec-10-k-filings-sentiment-analysis

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 1. Download 10-K filings
python src/ingest_edgar.py --tickers-file tickers.txt

# 2. Parse narrative sections into parquet
python src/parse_sections.py

# 3. Sentiment analysis with FinBERT
python src/sentiment_finbert.py

# 4. Merge filings with stock prices
python src/join_prices.py

# 5. Build features dataset
python src/build_features.py


Tools & Technologies

Python 3.10+

Libraries: duckdb, pandas, transformers, torch, yfinance, beautifulsoup4, tqdm

NLP Model: FinBERT (ProsusAI/finbert)

Visualization: Streamlit, Matplotlib, Plotly

Experiment Tracking: MLflow

Deployment: Docker

Version Control: Git + GitHub

Companies Covered

48 large-cap companies with consistent 10-K filings (2020–2024), across Technology, Healthcare, Consumer, Energy, and Industrials. Examples:
AAPL, MSFT, AMZN, NVDA, TSLA, ORCL, INTC, CSCO, JNJ, MRK, ABBV, UNH, CVS, COST, HD, NKE, MCD, SBUX, XOM, CVX, BA, LMT, UPS, FDX, ...

Author

HtayNaing Oo
Graduate Student (Youngstown State University)
Department of ComputerScience and Information Systems