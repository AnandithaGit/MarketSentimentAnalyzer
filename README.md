# AI-Based Market Sentiment and Stock Movement Analyzer

An NLP-based financial analysis project that uses FinBERT to analyze Microsoft-related financial news and investigates the relationship between news sentiment and stock returns.

## Overview

Financial news can influence investor expectations and market behavior. This project combines natural language processing with historical stock market data to study whether the sentiment expressed in financial news is associated with subsequent stock-price movements.

The system collects news articles using Finnhub, analyzes their sentiment using FinBERT, retrieves stock data using Yahoo Finance, and presents the results through an interactive Streamlit dashboard.

## Objectives

- Collect financial news related to Microsoft (MSFT).
- Analyze news sentiment using a pretrained financial language model.
- Retrieve historical Microsoft stock prices and calculate daily returns.
- Examine the relationship between news sentiment and stock returns using Pearson correlation.
- Visualize the findings through an interactive dashboard.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Financial news | Finnhub API |
| Sentiment analysis | ProsusAI/finbert |
| Stock market data | yfinance |
| Data processing | pandas |
| Statistical analysis | SciPy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |

## System Workflow

1. **News collection:** Retrieve Microsoft-related news from Finnhub over the selected analysis period.
2. **Data preparation:** Remove duplicate articles and reuse previously calculated sentiment scores where available.
3. **Sentiment analysis:** Process article headlines and summaries with FinBERT to obtain positive, negative, and neutral probabilities.
4. **Sentiment scoring:** Calculate a sentiment score as the positive probability minus the negative probability.
5. **Stock data collection:** Retrieve historical MSFT prices and calculate percentage returns.
6. **Date alignment:** Match daily news sentiment to a subsequent available trading date.
7. **Statistical analysis:** Calculate Pearson correlation and its p-value for the matched observations.
8. **Visualization:** Display the results through charts, metrics, and an interactive Streamlit dashboard.

## Sentiment Scoring

For each article:

`Sentiment Score = Positive Probability − Negative Probability`

The score ranges approximately from -1 to +1.

- Positive scores indicate relatively positive sentiment.
- Negative scores indicate relatively negative sentiment.
- Scores near zero indicate balanced sentiment or a relatively neutral result.

## Statistical Analysis

The project uses Pearson's correlation coefficient to measure the strength and direction of the linear relationship between daily sentiment scores and subsequent stock returns.

The p-value is used to assess whether the observed correlation is statistically significant at the 5% significance level.

**Important:** Correlation does not establish causation. Results may also be affected by article relevance, publication timing, market-wide events, and the number of matched observations.

## Dashboard Features

- Summary metrics for the analyzed news and market data
- News sentiment distribution
- Sentiment and stock-return visualizations
- Historical Microsoft stock-price charts
- Sentiment-versus-return scatter plot
- Searchable news explorer with sentiment information
- Downloadable article data
- Methodology and statistical interpretation

## Project Files

- `main.py` — News collection, sentiment analysis, stock data retrieval, and correlation analysis
- `app.py` — Interactive Streamlit dashboard
- `filter_microsoft_news.py` — Initial keyword-based Microsoft news filtering
- `sentiment_data.csv` — Cached article-level sentiment results
- `filtered_sentiment_data.csv` — Articles matching the initial keyword filter
- `stock_sentiment_analysis.csv` — Matched stock returns and daily sentiment observations
- `sentiment_correlation.png` — Sentiment-versus-return scatter plot

## Installation

### Prerequisites

- Python installed on your system
- A Finnhub API key
- Internet access for news and stock-data retrieval

### Install dependencies

```bash
pip install pandas requests torch transformers yfinance matplotlib scipy streamlit plotly
```

### Configure the API key

Add your Finnhub API key to the configuration in `main.py`.

Never commit API keys, passwords, or other secrets to GitHub. Use an environment variable or a local `.env` file excluded from version control.

### Run the analysis

```bash
python main.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Results and Interpretation

The analysis evaluates whether the sentiment of collected financial news is associated with subsequent Microsoft stock returns during the selected observation period.

Report the final article count, number of matched observations, correlation coefficient, p-value, and significance conclusion based on the latest validated run.

Do not interpret a statistically insignificant correlation as evidence that sentiment has no possible relationship with stock movements.

## Limitations

- Keyword-based filtering may include irrelevant articles or exclude relevant ones.
- Financial news may discuss multiple companies or broader market events.
- Mapping news to the next trading day is an approximation of publication-time effects.
- Pearson correlation measures linear association and does not establish causality.
- Results depend on the selected period and the number of matched observations.
- This project is for educational analysis and is not financial advice.

## Future Enhancements

- Improve article relevance classification.
- Compare different news-filtering strategies.
- Investigate publication-time-aware sentiment alignment.
- Evaluate alternative statistical and time-series methods.
- Add automated data refresh and historical performance comparisons.
- Deploy the dashboard for public access.

## Disclaimer

This project is intended for educational and research purposes only. It does not provide investment recommendations or guarantee future stock-price movements.
