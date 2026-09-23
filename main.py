from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import pandas as pd

company = "Microsoft"
ticker = "MSFT"

api = NewsApiClient(
    api_key="0c82ff7a70a644c386bef63cfd469abe"
)

model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

news = api.get_everything(
    q=f'"{company}" AND (stock OR shares OR earnings OR revenue OR profit OR investors OR financial)',
    language="en",
    sort_by="publishedAt",
    page_size=50
)

print("Company:", company)
print("Ticker:", ticker)
print("Total articles received:", len(news["articles"]))

financial_keywords = [
    "stock",
    "shares",
    "earnings",
    "revenue",
    "profit",
    "investor",
    "investors",
    "market",
    "financial",
    "forecast",
    "valuation",
    "dividend",
    "quarter",
    "sales",
    "stock price",
    "wall street",
    "analyst",
    "cloud",
    "azure",
    "artificial intelligence",
    "ai",
    "operating income",
    "cash flow",
    "guidance",
    "growth"
]

irrelevant_keywords = [
    "discount",
    "coupon",
    "fashion",
    "menswear",
    "recipe",
    "movie",
    "tv show",
    "gaming deal",
    "giveaway",
    "product review"
]

financial_articles = []

for article in news["articles"]:

    title = article["title"] or ""
    description = article["description"] or ""

    text = (title + " " + description).lower()

    if company.lower() not in text:
        continue

    has_financial_keyword = any(
        keyword in text
        for keyword in financial_keywords
    )

    has_irrelevant_keyword = any(
        keyword in text
        for keyword in irrelevant_keywords
    )

    if has_financial_keyword and not has_irrelevant_keyword:
        financial_articles.append(article)

print("Relevant financial articles:", len(financial_articles))

labels = [
    "positive",
    "negative",
    "neutral"
]

sentiment_counts = {
    "positive": 0,
    "negative": 0,
    "neutral": 0
}

sentiment_data = []

print("\n========================================")
print("        SENTIMENT ANALYSIS")
print("========================================")

for article in financial_articles:

    title = article["title"] or ""
    description = article["description"] or ""
    published_at = article["publishedAt"]

    text = title + " " + description

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.softmax(
        outputs.logits,
        dim=1
    )

    positive_prob = scores[0][0].item()
    negative_prob = scores[0][1].item()
    neutral_prob = scores[0][2].item()

    result = labels[
        torch.argmax(scores).item()
    ]

    confidence = torch.max(scores).item()

    sentiment_value = (
        positive_prob - negative_prob
    )

    article_date = pd.to_datetime(
        published_at
    ).date()

    sentiment_data.append({
        "Date": article_date,
        "Sentiment": sentiment_value
    })

    sentiment_counts[result] += 1

    print("\nTITLE:", title)
    print("DATE:", article_date)
    print("SENTIMENT:", result)
    print("SENTIMENT VALUE:", round(sentiment_value, 4))
    print("-" * 60)

if not sentiment_data:

    print("\nNo relevant financial articles found.")
    exit()

sentiment_df = pd.DataFrame(
    sentiment_data
)

daily_sentiment = (
    sentiment_df
    .groupby("Date")["Sentiment"]
    .mean()
    .reset_index()
)

daily_sentiment.columns = [
    "Date",
    "Daily Sentiment"
]

print("\n========================================")
print("        DAILY SENTIMENT")
print("========================================")

print(daily_sentiment.to_string(index=False))

start_date = (
    daily_sentiment["Date"].min()
    .strftime("%Y-%m-%d")
)

end_date = (
    daily_sentiment["Date"].max()
    .strftime("%Y-%m-%d")
)

stock = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

if stock.empty:

    print("\nNo stock data found.")
    exit()

if isinstance(stock.columns, pd.MultiIndex):

    stock.columns = stock.columns.get_level_values(0)

stock = stock.reset_index()

stock["Date"] = pd.to_datetime(
    stock["Date"]
).dt.date

stock["Daily Return"] = (
    stock["Close"].pct_change() * 100
)

stock_data = stock[
    ["Date", "Close", "Daily Return"]
]

merged_data = pd.merge(
    daily_sentiment,
    stock_data,
    on="Date",
    how="inner"
)

print("\n========================================")
print("        SENTIMENT VS STOCK MOVEMENT")
print("========================================")

if merged_data.empty:

    print("\nNo matching dates found.")

else:

    print(
        merged_data.to_string(index=False)
    )

    if len(merged_data) >= 2:

        correlation = merged_data[
            "Daily Sentiment"
        ].corr(
            merged_data["Daily Return"]
        )

        print("\n========================================")
        print("        CORRELATION")
        print("========================================")

        print(
            "Sentiment-Stock Correlation:",
            round(correlation, 4)
        )

    else:

        print(
            "\nNot enough matching dates to calculate correlation."
        )

print("\n========================================")
print("        OVERALL SENTIMENT")
print("========================================")

print("Positive:", sentiment_counts["positive"])
print("Negative:", sentiment_counts["negative"])
print("Neutral:", sentiment_counts["neutral"])

overall_sentiment = (
    sum(sentiment_df["Sentiment"])
    / len(sentiment_df)
)

print(
    "Overall Sentiment Score:",
    round(overall_sentiment, 4)
)

if overall_sentiment > 0.05:

    overall_label = "POSITIVE"

elif overall_sentiment < -0.05:

    overall_label = "NEGATIVE"

else:

    overall_label = "NEUTRAL"

print(
    "Overall Sentiment:",
    overall_label
)