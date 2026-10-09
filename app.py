
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

st.set_page_config(
    page_title="MarketPulse | Sentiment Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_FILE = os.path.join(BASE_DIR, "sentiment_data.csv")
ANALYSIS_FILE = os.path.join(BASE_DIR, "stock_sentiment_analysis.csv")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1018;
}

[data-testid="stHeader"] {
    background: rgba(11, 16, 24, 0.95);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111925;
    border-right: 1px solid #263244;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #151f2e, #101722);
    border: 1px solid #28364a;
    padding: 20px 22px;
    border-radius: 14px;
    min-height: 125px;
}

[data-testid="stMetricLabel"] {
    color: #9eafc4;
    font-size: 0.88rem;
}

[data-testid="stMetricValue"] {
    color: #f2f6fc;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
}

[data-testid="stMetricDelta"] {
    font-size: 0.85rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.035em;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 700;
}

h2 {
    font-size: 1.5rem !important;
}

.eyebrow {
    color: #62d6b5;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.subtitle {
    color: #9eafc4;
    font-size: 1rem;
    margin-top: -0.5rem;
}

.section-note {
    color: #91a1b5;
    font-size: 0.88rem;
}

.status-card {
    background: #131e2b;
    border: 1px solid #29384b;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 12px 0 24px 0;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

hr {
    border-color: #283447;
}

.stDownloadButton button {
    border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not os.path.exists(NEWS_FILE):
        raise FileNotFoundError("sentiment_data.csv was not found.")

    if not os.path.exists(ANALYSIS_FILE):
        raise FileNotFoundError("stock_sentiment_analysis.csv was not found.")

    news = pd.read_csv(NEWS_FILE)
    analysis = pd.read_csv(ANALYSIS_FILE)

    required_news = {"Date", "Title", "Sentiment", "URL"}
    required_analysis = {"Date", "Close", "Return", "Sentiment"}

    if not required_news.issubset(news.columns):
        raise ValueError("The news CSV is missing required columns.")

    if not required_analysis.issubset(analysis.columns):
        raise ValueError("The analysis CSV is missing required columns.")

    news["Date"] = pd.to_datetime(news["Date"], errors="coerce")
    analysis["Date"] = pd.to_datetime(analysis["Date"], errors="coerce")

    for column in ["Sentiment", "Positive", "Negative", "Neutral"]:
        if column in news.columns:
            news[column] = pd.to_numeric(news[column], errors="coerce")

    for column in ["Close", "Return", "Sentiment"]:
        analysis[column] = pd.to_numeric(
            analysis[column], errors="coerce"
        )

    news = news.dropna(subset=["Date", "Sentiment", "Title"])
    analysis = analysis.dropna(
        subset=["Date", "Close", "Return", "Sentiment"]
    )

    news["DateOnly"] = news["Date"].dt.date
    analysis["DateOnly"] = analysis["Date"].dt.date

    return news, analysis


def make_chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color="#dce6f2",
            size=12
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hoverlabel=dict(
            bgcolor="#172334",
            font_color="#ffffff"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#344155",
        tickfont=dict(color="#9eafc4")
    )

    fig.update_yaxes(
        gridcolor="#263244",
        zerolinecolor="#42516a",
        tickfont=dict(color="#9eafc4")
    )

    return fig


st.markdown('<p class="eyebrow">FINANCIAL INTELLIGENCE · NLP ANALYTICS</p>',
            unsafe_allow_html=True)

st.title("Market Sentiment Analyzer")

st.markdown(
    '<p class="subtitle">Understanding the relationship between financial news '
    'sentiment and Microsoft stock performance.</p>',
    unsafe_allow_html=True
)

try:
    news, analysis = load_data()
except Exception as error:
    st.error(str(error))
    st.info(
        "Keep app.py, sentiment_data.csv, and "
        "stock_sentiment_analysis.csv in the same project folder."
    )
    st.stop()

if news.empty or analysis.empty:
    st.error("The CSV files contain no usable observations.")
    st.stop()

min_date = min(news["DateOnly"].min(), analysis["DateOnly"].min())
max_date = max(news["DateOnly"].max(), analysis["DateOnly"].max())

st.sidebar.markdown('<p class="eyebrow">WORKSPACE</p>', unsafe_allow_html=True)
st.sidebar.title("Analysis Controls")
st.sidebar.caption("Microsoft Corporation · NASDAQ: MSFT")

selected_dates = st.sidebar.date_input(
    "Analysis period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
    start_date = selected_dates[0]
    end_date = selected_dates[0]
else:
    start_date, end_date = min_date, max_date

filtered_news = news[
    (news["DateOnly"] >= start_date) &
    (news["DateOnly"] <= end_date)
].copy()

filtered_analysis = analysis[
    (analysis["DateOnly"] >= start_date) &
    (analysis["DateOnly"] <= end_date)
].copy()

st.sidebar.divider()
st.sidebar.metric("Articles in period", f"{len(filtered_news):,}")
st.sidebar.metric("Matched observations", f"{len(filtered_analysis):,}")
st.sidebar.caption("Data loaded from your saved CSV files.")

if filtered_news.empty:
    st.warning("No news articles exist in the selected period.")

if filtered_analysis.empty:
    st.warning("No matched stock observations exist in this period.")
    st.stop()

filtered_analysis = filtered_analysis.sort_values("Date")
filtered_news = filtered_news.sort_values("Date", ascending=False)

average_sentiment = filtered_news["Sentiment"].mean()
average_return = filtered_analysis["Return"].mean()
latest_close = filtered_analysis["Close"].iloc[-1]

if average_sentiment > 0.05:
    sentiment_label = "Positive"
elif average_sentiment < -0.05:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

if len(filtered_analysis) >= 3 and filtered_analysis["Sentiment"].nunique() > 1 and filtered_analysis["Return"].nunique() > 1:
    correlation, p_value = pearsonr(
        filtered_analysis["Sentiment"],
        filtered_analysis["Return"]
    )
else:
    correlation, p_value = np.nan, np.nan

st.markdown(
    f'<div class="status-card"><span class="eyebrow">CURRENT VIEW</span><br>'
    f'<b>{start_date.strftime("%d %b %Y")} — '
    f'{end_date.strftime("%d %b %Y")}</b>'
    f'<br><span class="section-note">Saved historical analysis · '
    f'Microsoft (MSFT)</span></div>',
    unsafe_allow_html=True
)

st.subheader("Performance Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Articles Analyzed", f"{len(filtered_news):,}")
m2.metric("Average Sentiment", f"{average_sentiment:+.4f}", sentiment_label)
m3.metric(
    "Pearson Correlation",
    f"{correlation:.4f}" if pd.notna(correlation) else "N/A"
)
m4.metric("Latest Matched Close", f"${latest_close:,.2f}")

st.caption(
    f"Average matched next-day stock return: {average_return:+.3f}%."
    " These are descriptive statistics, not investment recommendations."
)

st.divider()

overview_tab, markets_tab, news_tab, methodology_tab = st.tabs([
    "Overview",
    "Market Performance",
    "News Explorer",
    "Methodology"
])

with overview_tab:
    st.subheader("Sentiment Composition")
    st.markdown(
        '<p class="section-note">Distribution of article-level FinBERT scores.</p>',
        unsafe_allow_html=True
    )

    positive_count = int((filtered_news["Sentiment"] > 0.05).sum())
    negative_count = int((filtered_news["Sentiment"] < -0.05).sum())
    neutral_count = len(filtered_news) - positive_count - negative_count

    sentiment_df = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Articles": [positive_count, neutral_count, negative_count]
    })

    left, right = st.columns([0.9, 1.4])

    with left:
        fig_donut = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Articles",
            hole=0.66,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#62d6b5",
                "Neutral": "#718096",
                "Negative": "#ff7b8a"
            }
        )

        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            hole=0.66,
            marker=dict(line=dict(color="#0b1018", width=3))
        )

        fig_donut.add_annotation(
            text=f"<b>{len(filtered_news):,}</b><br><sup>Articles</sup>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="#f2f6fc")
        )

        make_chart_layout(fig_donut, 350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        fig_bar = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Articles",
            color="Sentiment",
            text="Articles",
            color_discrete_map={
                "Positive": "#62d6b5",
                "Neutral": "#718096",
                "Negative": "#ff7b8a"
            }
        )

        fig_bar.update_traces(
            textposition="outside",
            marker_line_width=0
        )

        fig_bar.update_layout(showlegend=False)
        make_chart_layout(fig_bar, 350)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    st.subheader("Sentiment vs. Next Trading-Day Return")

    fig_scatter = px.scatter(
        filtered_analysis,
        x="Sentiment",
        y="Return",
        hover_data=["Date", "Close"],
        labels={
            "Sentiment": "Daily news sentiment score",
            "Return": "Next trading-day return (%)"
        }
    )

    fig_scatter.update_traces(
        marker=dict(
            size=10,
            color="#72b7ff",
            line=dict(width=1, color="#dce6f2")
        )
    )

    if len(filtered_analysis) >= 2 and filtered_analysis["Sentiment"].nunique() >= 2:
        x_values = filtered_analysis["Sentiment"].to_numpy()
        y_values = filtered_analysis["Return"].to_numpy()
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(x_values.min(), x_values.max(), 100)

        fig_scatter.add_trace(
            go.Scatter(
                x=line_x,
                y=slope * line_x + intercept,
                mode="lines",
                name="Linear trend",
                line=dict(color="#62d6b5", width=2, dash="dash")
            )
        )

    make_chart_layout(fig_scatter, 430)
    st.plotly_chart(fig_scatter, use_container_width=True)

    if pd.notna(correlation):
        c1, c2 = st.columns(2)

        c1.metric("Pearson r", f"{correlation:.4f}")
        c2.metric("P-value", f"{p_value:.4f}")

        if p_value < 0.05:
            st.success(
                "The observed linear correlation is statistically significant "
                "at the 5% level. This does not establish causation."
            )
        else:
            st.info(
                "The observed linear correlation is not statistically "
                "significant at the 5% level."
            )

with markets_tab:
    st.subheader("Microsoft Stock Performance")

    price_chart = px.line(
        filtered_analysis,
        x="Date",
        y="Close",
        markers=True,
        labels={
            "Date": "Trading date",
            "Close": "Adjusted close (USD)"
        }
    )

    price_chart.update_traces(
        line=dict(color="#72b7ff", width=3),
        marker=dict(size=5)
    )

    make_chart_layout(price_chart, 430)
    st.plotly_chart(price_chart, use_container_width=True)

    st.subheader("Daily Stock Returns")

    return_chart = px.bar(
        filtered_analysis,
        x="Date",
        y="Return",
        labels={
            "Date": "Trading date",
            "Return": "Return (%)"
        },
        color="Return",
        color_continuous_scale=["#ff7b8a", "#263244", "#62d6b5"],
        color_continuous_midpoint=0
    )

    return_chart.update_layout(coloraxis_showscale=False)
    make_chart_layout(return_chart, 350)
    st.plotly_chart(return_chart, use_container_width=True)

    st.subheader("Matched Market Data")

    market_table = filtered_analysis[
        ["Date", "Close", "Return", "Sentiment"]
    ].copy()

    market_table["Close"] = market_table["Close"].round(2)
    market_table["Return"] = market_table["Return"].round(4)
    market_table["Sentiment"] = market_table["Sentiment"].round(4)

    st.dataframe(
        market_table,
        use_container_width=True,
        hide_index=True
    )

with news_tab:
    st.subheader("Article Explorer")
    st.markdown(
        '<p class="section-note">Inspect saved news records and FinBERT scores.</p>',
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Search article headlines",
        placeholder="Try: Microsoft, AI, cloud, earnings..."
    )

    sentiment_filter = st.multiselect(
        "Sentiment category",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

    display_news = filtered_news.copy()

    if search_text:
        display_news = display_news[
            display_news["Title"].str.contains(
                search_text, case=False, na=False
            )
        ]

    display_news["Category"] = np.select(
        [
            display_news["Sentiment"] > 0.05,
            display_news["Sentiment"] < -0.05
        ],
        ["Positive", "Negative"],
        default="Neutral"
    )

    display_news = display_news[
        display_news["Category"].isin(sentiment_filter)
    ]

    display_columns = [
        column for column in [
            "Date", "Title", "Source", "Category", "Sentiment",
            "Positive", "Negative", "Neutral", "URL"
        ]
        if column in display_news.columns
    ]

    st.write(f"Showing **{len(display_news):,}** articles.")

    st.dataframe(
        display_news[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("Article URL"),
            "Sentiment": st.column_config.NumberColumn(
                "Sentiment score", format="%.4f"
            ),
            "Positive": st.column_config.NumberColumn(format="%.4f"),
            "Negative": st.column_config.NumberColumn(format="%.4f"),
            "Neutral": st.column_config.NumberColumn(format="%.4f")
        }
    )

    download_data = display_news.drop(
        columns=["DateOnly", "Category"], errors="ignore"
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered articles",
        data=download_data,
        file_name="filtered_sentiment_articles.csv",
        mime="text/csv"
    )

with methodology_tab:
    st.subheader("How the Analyzer Works")

    st.markdown("""
    **1. Financial news collection**

    News articles are collected through Finnhub and stored in a local CSV
    cache to reduce repeated sentiment inference.

    **2. Natural language processing**

    ProsusAI/finbert classifies each article's financial sentiment into
    positive, negative, and neutral probabilities.

    **3. Sentiment score**

    The article sentiment score is calculated as:

    `Positive probability - Negative probability`

    Scores above 0 indicate a positive lean; scores below 0 indicate a
    negative lean. Scores near 0 indicate a more balanced result.

    **4. Stock market data**

    Adjusted closing prices are collected using yfinance. Daily percentage
    returns are calculated from consecutive trading sessions.

    **5. Temporal alignment**

    Daily news sentiment is assigned to the next available trading day in
    the current pipeline. Multiple news dates mapped to one trading day
    are averaged.

    **6. Statistical analysis**

    Pearson's correlation coefficient measures linear association between
    matched sentiment and returns. The p-value is used to assess statistical
    significance at the 5% level.
    """)

    st.warning(
        "Limitations: the number of matched trading days is much smaller than "
        "the number of articles. The current date-matching method does not "
        "fully account for each article's publication time. Correlation does "
        "not demonstrate causation or predict future prices."
    )

    if pd.notna(correlation):
        st.markdown("#### Results for the selected period")

        result_table = pd.DataFrame({
            "Metric": [
                "Articles included",
                "Matched observations",
                "Average sentiment",
                "Average matched stock return (%)",
                "Pearson correlation",
                "P-value"
            ],
            "Value": [
                f"{len(filtered_news):,}",
                f"{len(filtered_analysis):,}",
                f"{average_sentiment:.4f}",
                f"{average_return:.4f}",
                f"{correlation:.4f}",
                f"{p_value:.4f}"
            ]
        })

        st.dataframe(result_table, hide_index=True, use_container_width=True)

st.divider()

st.markdown(
    '<p class="section-note">Market Sentiment Analyzer · '
    'Finnhub · FinBERT · yfinance · Streamlit</p>',
    unsafe_allow_html=True
)
