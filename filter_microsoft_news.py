
import os
import re
import pandas as pd

INPUT_FILE = "sentiment_data.csv"
OUTPUT_FILE = "filtered_sentiment_data.csv"

RELEVANCE_TERMS = [
    "microsoft",
    "msft",
    "azure",
    "microsoft 365",
    "office 365",
    "windows",
    "xbox",
    "copilot",
    "github",
    "linkedin",
    "bing",
    "activision blizzard",
    "nuance communications"
]

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. Run this script from your project folder."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "Date", "Title", "Summary", "Source",
    "URL", "Sentiment", "Positive", "Negative", "Neutral"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

df["Title"] = df["Title"].fillna("").astype(str)
df["Summary"] = df["Summary"].fillna("").astype(str)

def get_relevance_reason(row):
    title = row["Title"].lower()
    summary = row["Summary"].lower()

    if re.search(r"\bmicrosoft\b|\bmsft\b", title):
        return "Microsoft mentioned in title"

    if any(term in title for term in RELEVANCE_TERMS[2:]):
        return "Microsoft product mentioned in title"

    if re.search(r"\bmicrosoft\b|\bmsft\b", summary):
        return "Microsoft mentioned in summary"

    if any(term in summary for term in RELEVANCE_TERMS[2:]):
        return "Microsoft product mentioned in summary"

    return ""

df["RelevanceReason"] = df.apply(get_relevance_reason, axis=1)

filtered = df[df["RelevanceReason"] != ""].copy()
filtered = filtered.drop_duplicates(subset=["URL"], keep="last")

print("\n========== MICROSOFT NEWS FILTER ==========")
print(f"Original articles: {len(df)}")
print(f"Matching articles: {len(filtered)}")
print(f"Excluded articles: {len(df) - len(filtered)}")

if len(df) > 0:
    print(f"Retention rate: {len(filtered) / len(df) * 100:.2f}%")

if not filtered.empty:
    filtered.to_csv(OUTPUT_FILE, index=False)

    print("\nMatches by reason:")
    print(filtered["RelevanceReason"].value_counts().to_string())

    print("\nSample matching headlines:")

    for title in filtered["Title"].head(10):
        print(f"- {title}")

    print(f"\nSaved filtered data to {OUTPUT_FILE}")
else:
    print("\nNo matching articles found.")
    print("No output CSV was created.")

print("===========================================\n")
