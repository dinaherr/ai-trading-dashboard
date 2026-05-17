import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime
import sqlite3
import os
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("AI Trading Research Dashboard")
st.caption("Phase 1 + Phase 2: Watchlist, charts, indicators, scoring, AI summaries, news sentiment")

# ── Database setup ────────────────────────────────────────────────────────────
DB_PATH = "data/trades.db"
os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT,
            ticker TEXT,
            action TEXT,
            entry_price REAL,
            ai_score INTEGER,
            rsi REAL,
            ma20 REAL,
            ma50 REAL,
            signal TEXT,
            notes TEXT,
            status TEXT DEFAULT 'Open'
        )
    """)
    conn.commit()
    conn.close()

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()
    return df

def save_trade(entry_date, ticker, action, entry_price, ai_score, rsi, ma20, ma50, signal, notes):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (entry_date, ticker, action, entry_price, ai_score, rsi, ma20, ma50, signal, notes, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open')
    """, (entry_date, ticker, action, entry_price, ai_score, rsi, ma20, ma50, signal, notes))
    conn.commit()
    conn.close()

def close_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE trades SET status = 'Closed' WHERE id = ?", (trade_id,))
    conn.commit()
    conn.close()

def delete_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    conn.commit()
    conn.close()

init_db()

# ── API key validation ────────────────────────────────────────────────────────
openai_key   = st.secrets.get("OPENAI_API_KEY", None)
alpha_key    = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
finnhub_key  = st.secrets.get("FINNHUB_API_KEY", None)
sec_agent    = st.secrets.get("SEC_USER_AGENT", "MyApp myemail@email.com")

if not openai_key:
    st.sidebar.warning("OpenAI key missing — AI summaries disabled")
    client = None
else:
    client = OpenAI(api_key=openai_key)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Watchlist")
    tickers_input = st.text_input("Enter tickers (comma separated)", "NVDA, CRWD, PANW, AMD, SPY")
    period = st.selectbox("Time period", ["3mo", "6mo", "1y", "2y"], index=1)
    selected = st.selectbox(
        "Deep-dive ticker",
        [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    )

    st.divider()
    st.header("Phase 2")
    st.caption("✅ News sentiment")
    st.caption("🔒 SEC insider trades (next)")
    st.caption("🔒 Finnhub insider trades (next)")
    st.caption("🔒 Politician trades")
    st.caption("🔒 FDA catalysts")
    st.caption("🔒 Earnings transcripts")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ── Helper functions ──────────────────────────────────────────────────────────
def safe_float(val):
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    return float(val)

def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = -delta.where(delta < 0, 0).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df["MA20"]      = df["Close"].rolling(20).mean()
    df["MA50"]      = df["Close"].rolling(50).mean()
    df["RSI"]       = calculate_rsi(df)
    df["VolumeAvg"] = df["Volume"].rolling(20).mean()
    return df.dropna()

def score_stock(df):
    score   = 50
    reasons = []
    latest  = df.iloc[-1]

    price   = safe_float(latest["Close"])
    ma20    = safe_float(latest["MA20"])
    ma50    = safe_float(latest["MA50"])
    rsi     = safe_float(latest["RSI"])
    vol     = safe_float(latest["Volume"])
    avg_vol = safe_float(latest["VolumeAvg"])

    if price > ma50:
        score += 20
        reasons.append("Price above 50-day MA (+20)")
    else:
        score -= 20
        reasons.append("Price below 50-day MA (-20)")

    if ma20 > ma50:
        score += 15
        reasons.append("20-day MA above 50-day MA — bullish cross (+15)")

    if 45 <= rsi <= 70:
        score += 15
        reasons.append(f"RSI {rsi:.1f} in healthy range 45–70 (+15)")
    elif rsi > 75:
        score -= 15
        reasons.append(f"RSI {rsi:.1f} overbought >75 (-15)")
    else:
        reasons.append(f"RSI {rsi:.1f} outside ideal range (no change)")

    if vol > avg_vol:
        score += 10
        reasons.append("Volume above 20-day average (+10)")
    else:
        reasons.append("Volume below average (no change)")

    return max(0, min(100, int(score))), reasons

def get_signal(score):
    if score >= 70:
        return "Bullish"
    elif score >= 50:
        return "Neutral"
    return "Bearish"

# ── Phase 2: News sentiment ───────────────────────────────────────────────────
@st.cache_data(ttl=1800)  # cache 30 mins — AV has rate limits
def get_news_sentiment(ticker):
    if not alpha_key:
        return None, "Alpha Vantage key missing"

    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&limit=20"
            f"&apikey={alpha_key}"
        )
        response = requests.get(url, timeout=10)
        data = response.json()

        # Rate limit check
        if "Information" in data or "Note" in data:
            msg = data.get("Information") or data.get("Note")
            return None, f"Rate limit hit: {msg}"

        if "feed" not in data or len(data["feed"]) == 0:
            return None, "No news found for this ticker"

        articles     = data["feed"]
        scores       = []
        top_articles = []

        for article in articles:
            # Find sentiment score specific to this ticker
            ticker_sentiments = article.get("ticker_sentiment", [])
            for ts in ticker_sentiments:
                if ts.get("ticker") == ticker:
                    try:
                        scores.append(float(ts["ticker_sentiment_score"]))
                    except:
                        pass

            # Collect top 3 headlines
            if len(top_articles) < 3:
                top_articles.append({
                    "title":     article.get("title", "No title"),
                    "source":    article.get("source", "Unknown"),
                    "url":       article.get("url", ""),
                    "time":      article.get("time_published", "")[:8],
                    "sentiment": article.get("overall_sentiment_label", "Neutral")
                })

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score >= 0.15:
            sentiment_label = "Bullish"
        elif avg_score <= -0.15:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        return {
            "article_count":     len(articles),
            "avg_score":         round(avg_score, 4),
            "sentiment_label":   sentiment_label,
            "top_articles":      top_articles,
            "scored_articles":   len(scores)
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except requests.exceptions.ConnectionError:
        return None, "Connection error — check your internet"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ── AI analysis ───────────────────────────────────────────────────────────────
def generate_ai_analysis(ticker, latest, score, sentiment_data=None):
    sentiment_block = ""
    if sentiment_data:
        sentiment_block = f"""
    News Sentiment Score: {sentiment_data['avg_score']}
    News Sentiment Label: {sentiment_data['sentiment_label']}
    Articles Analyzed: {sentiment_data['article_count']}
    """

    prompt = f"""
    You are an AI stock research assistant for paper trading only.
    Analyze this ticker using the provided technical indicators and news sentiment.

    Ticker: {ticker}
    Latest Close: {safe_float(latest['Close']):.2f}
    RSI: {safe_float(latest['RSI']):.1f}
    Volume: {safe_float(latest['Volume']):.0f}
    20-day Moving Average: {safe_float(latest['MA20']):.2f}
    50-day Moving Average: {safe_float(latest['MA50']):.2f}
    Score: {score}/100
    {sentiment_block}

    Please explain:
    1. Whether this looks bullish, neutral, or weak and why
    2. What the moving averages suggest about the trend
    3. What the RSI level means right now
    4. Whether volume supports or weakens the move
    5. What the news sentiment suggests (if available)
    6. One specific thing a paper trader should watch for next

    Keep it beginner-friendly. Be specific to this ticker's numbers.
    End with a disclaimer that this is not financial advice.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain stock setups clearly for beginner paper traders. Be specific, concise, and use plain English."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

@st.cache_data(ttl=300)
def get_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_current_price(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except:
        return None

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Watchlist Overview", "Deep Dive", "Paper Trade Log"])

# ── TAB 1: Watchlist overview ─────────────────────────────────────────────────
with tab1:
    st.subheader("Watchlist Summary")
    summary_rows = []
    all_data     = {}

    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty:
            continue
        data = add_indicators(raw.copy())
        if data.empty:
            continue
        all_data[ticker] = data
        latest = data.iloc[-1]
        score, _ = score_stock(data)
        signal   = get_signal(score)
        summary_rows.append({
            "Ticker": ticker,
            "Close":  f"${safe_float(latest['Close']):.2f}",
            "RSI":    f"{safe_float(latest['RSI']):.1f}",
            "MA20":   f"${safe_float(latest['MA20']):.2f}",
            "MA50":   f"${safe_float(latest['MA50']):.2f}",
            "Score":  f"{score}/100",
            "Signal": signal
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df["_sort"] = summary_df["Score"].str.replace("/100", "").astype(int)
        summary_df = summary_df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    for ticker in tickers:
        if ticker not in all_data:
            st.error(f"No data for {ticker}")
            continue

        data   = all_data[ticker]
        latest = data.iloc[-1]
        score, reasons = score_stock(data)
        signal = get_signal(score)

        st.subheader(ticker)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close",    f"${safe_float(latest['Close']):.2f}")
        col2.metric("RSI",      f"{safe_float(latest['RSI']):.1f}")
        col3.metric("Volume",   f"{safe_float(latest['Volume']):,.0f}")
        col4.metric("AI Score", f"{score}/100")

        if signal == "Bullish":
            st.success("Bullish — strong setup")
        elif signal == "Neutral":
            st.info("Neutral — watch carefully")
        else:
            st.warning("Weak setup — caution")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(color="royalblue")))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False, title=f"{ticker} — {period} chart")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Score breakdown"):
            for r in reasons:
                st.write(f"• {r}")

        with st.expander("AI Assistant Analysis"):
            if client is None:
                st.warning("Add your OPENAI_API_KEY in Streamlit Secrets to enable this.")
            else:
                if st.button(f"Generate AI Analysis for {ticker}", key=f"ai_{ticker}"):
                    with st.spinner(f"Analyzing {ticker}..."):
                        analysis = generate_ai_analysis(ticker, latest, score)
                        st.write(analysis)

        st.divider()

# ── TAB 2: Deep dive + news sentiment ────────────────────────────────────────
with tab2:
    st.subheader(f"Deep Dive: {selected}")
    raw = get_data(selected, period)

    if raw.empty:
        st.error(f"No data for {selected}")
    else:
        data   = add_indicators(raw.copy())
        latest = data.iloc[-1]
        score, reasons = score_stock(data)
        signal = get_signal(score)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score",  f"{score}/100")
        col2.metric("Signal", signal)
        col3.metric("RSI",    f"{safe_float(latest['RSI']):.1f}")
        col4.metric("Close",  f"${safe_float(latest['Close']):.2f}")

        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price"))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(color="orange")))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(color="royalblue")))
        fig2.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected} — Deep Dive")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Score breakdown")
        for r in reasons:
            st.write(f"• {r}")

        # ── News sentiment expander ───────────────────────────────────────────
        with st.expander("News Sentiment", expanded=True):
            if not alpha_key:
                st.warning("Add ALPHA_VANTAGE_API_KEY to Streamlit Secrets to enable news sentiment.")
            else:
                with st.spinner(f"Fetching news sentiment for {selected}..."):
                    sentiment_data, error = get_news_sentiment(selected)

                if error:
                    st.warning(f"Could not load sentiment: {error}")
                    sentiment_data = None
                else:
                    s_col1, s_col2, s_col3 = st.columns(3)
                    s_col1.metric("Sentiment",      sentiment_data["sentiment_label"])
                    s_col2.metric("Avg Score",      sentiment_data["avg_score"])
                    s_col3.metric("Articles Found", sentiment_data["article_count"])

                    if sentiment_data["sentiment_label"] == "Bullish":
                        st.success("News sentiment is Bullish for this ticker")
                    elif sentiment_data["sentiment_label"] == "Bearish":
                        st.error("News sentiment is Bearish for this ticker")
                    else:
                        st.info("News sentiment is Neutral for this ticker")

                    st.markdown("**Top recent headlines:**")
                    for article in sentiment_data["top_articles"]:
                        date_str = article['time']
                        if len(date_str) == 8:
                            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        st.markdown(
                            f"- [{article['title']}]({article['url']})  "
                            f"*{article['source']} · {date_str} · {article['sentiment']}*"
                        )

        # ── AI summary ────────────────────────────────────────────────────────
        st.subheader("AI Summary")
        if client is None:
            st.warning("Add your OPENAI_API_KEY in Streamlit Secrets to enable this.")
        elif st.button("Generate AI Summary", key="ai_deepdive"):
            with st.spinner("Analyzing..."):
                sent, _ = get_news_sentiment(selected) if alpha_key else (None, None)
                analysis = generate_ai_analysis(selected, latest, score, sentiment_data=sent)
                st.info(analysis)

# ── TAB 3: Paper trade log ────────────────────────────────────────────────────
with tab3:
    st.subheader("Paper Trade Log")
    st.caption("Trades are saved permanently. No real money involved.")

    with st.expander("Log a new trade"):
        c1, c2, c3 = st.columns(3)
        trade_ticker = c1.text_input("Ticker", value=selected)
        trade_action = c2.selectbox("Action", ["BUY", "SELL"])

        raw_log = get_data(trade_ticker, "5d")
        default_price = float(raw_log["Close"].iloc[-1]) if not raw_log.empty else 100.0
        trade_price = c3.number_input("Entry Price", value=default_price)

        raw_ind = get_data(trade_ticker, period)
        if not raw_ind.empty:
            ind_data        = add_indicators(raw_ind.copy())
            ind_latest      = ind_data.iloc[-1]
            trade_score, _  = score_stock(ind_data)
            trade_signal    = get_signal(trade_score)
            trade_rsi       = safe_float(ind_latest["RSI"])
            trade_ma20      = safe_float(ind_latest["MA20"])
            trade_ma50      = safe_float(ind_latest["MA50"])
        else:
            trade_score  = 0
            trade_signal = "Unknown"
            trade_rsi    = 0.0
            trade_ma20   = 0.0
            trade_ma50   = 0.0

        trade_notes = st.text_input("Notes (optional)")
        st.caption(f"Will save with: Score {trade_score}/100 | RSI {trade_rsi:.1f} | Signal {trade_signal}")

        if st.button("Log Trade"):
            save_trade(
                entry_date  = datetime.now().strftime("%Y-%m-%d %H:%M"),
                ticker      = trade_ticker,
                action      = trade_action,
                entry_price = trade_price,
                ai_score    = trade_score,
                rsi         = trade_rsi,
                ma20        = trade_ma20,
                ma50        = trade_ma50,
                signal      = trade_signal,
                notes       = trade_notes
            )
            st.success(f"Logged {trade_action} {trade_ticker} at ${trade_price:.2f}")
            st.rerun()

    trades_df = load_trades()

    if trades_df.empty:
        st.info("No trades logged yet. Use the form above to add one.")
    else:
        st.subheader("Open Trades")
        open_trades = trades_df[trades_df["status"] == "Open"].copy()

        if not open_trades.empty:
            perf_rows = []
            for _, row in open_trades.iterrows():
                current_price = get_current_price(row["ticker"])
                if current_price and row["entry_price"] > 0:
                    if row["action"] == "BUY":
                        gain_pct = ((current_price - row["entry_price"]) / row["entry_price"]) * 100
                    else:
                        gain_pct = ((row["entry_price"] - current_price) / row["entry_price"]) * 100
                    gain_str = f"{gain_pct:+.2f}%"
                else:
                    current_price = 0.0
                    gain_str = "N/A"

                entry_dt  = datetime.strptime(row["entry_date"], "%Y-%m-%d %H:%M")
                days_open = (datetime.now() - entry_dt).days

                perf_rows.append({
                    "ID":           row["id"],
                    "Date":         row["entry_date"],
                    "Ticker":       row["ticker"],
                    "Action":       row["action"],
                    "Entry $":      f"${row['entry_price']:.2f}",
                    "Current $":    f"${current_price:.2f}",
                    "Gain/Loss":    gain_str,
                    "Days Open":    days_open,
                    "Score@Entry":  f"{row['ai_score']}/100",
                    "RSI@Entry":    f"{row['rsi']:.1f}",
                    "Signal@Entry": row["signal"],
                    "Notes":        row["notes"]
                })

            perf_df = pd.DataFrame(perf_rows)
            st.dataframe(perf_df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

            st.subheader("Manage trades")
            trade_ids    = open_trades["id"].tolist()
            trade_labels = [
                f"{row['ticker']} {row['action']} @ ${row['entry_price']:.2f} ({row['entry_date']})"
                for _, row in open_trades.iterrows()
            ]
            selected_label = st.selectbox("Select a trade", trade_labels)
            selected_id    = trade_ids[trade_labels.index(selected_label)]

            mc1, mc2 = st.columns(2)
            if mc1.button("Mark as Closed"):
                close_trade(selected_id)
                st.success("Trade marked as closed.")
                st.rerun()
            if mc2.button("Delete Trade"):
                delete_trade(selected_id)
                st.success("Trade deleted.")
                st.rerun()

        closed_trades = trades_df[trades_df["status"] == "Closed"]
        if not closed_trades.empty:
            st.subheader("Closed Trades")
            st.dataframe(closed_trades[[
                "entry_date", "ticker", "action", "entry_price",
                "ai_score", "rsi", "signal", "notes"
            ]].rename(columns={
                "entry_date":  "Date",
                "ticker":      "Ticker",
                "action":      "Action",
                "entry_price": "Entry $",
                "ai_score":    "Score@Entry",
                "rsi":         "RSI@Entry",
                "signal":      "Signal@Entry",
                "notes":       "Notes"
            }), use_container_width=True, hide_index=True)
