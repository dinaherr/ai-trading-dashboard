import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime, date, timedelta
import sqlite3
import os
import requests

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("AI Trading Research Dashboard")
st.caption("Phase 1 + Phase 2: Watchlist, charts, indicators, scoring, news sentiment, insider trades")

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
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_name TEXT,
            usage_date TEXT,
            count INTEGER DEFAULT 0
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

# ── API request counter ───────────────────────────────────────────────────────
AV_DAILY_LIMIT      = 25
FINNHUB_DAILY_LIMIT = 60

def get_usage_today(api_name):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT count FROM api_usage WHERE api_name=? AND usage_date=?", (api_name, today))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_usage(api_name, amount=1):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT count FROM api_usage WHERE api_name=? AND usage_date=?", (api_name, today))
    row = c.fetchone()
    if row:
        c.execute("UPDATE api_usage SET count=count+? WHERE api_name=? AND usage_date=?", (amount, api_name, today))
    else:
        c.execute("INSERT INTO api_usage (api_name, usage_date, count) VALUES (?,?,?)", (api_name, today, amount))
    conn.commit()
    conn.close()

def requests_remaining(api_name, limit):
    return max(0, limit - get_usage_today(api_name))

# ── API keys ──────────────────────────────────────────────────────────────────
openai_key  = st.secrets.get("OPENAI_API_KEY", None)
alpha_key   = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
finnhub_key = st.secrets.get("FINNHUB_API_KEY", None)
sec_agent   = st.secrets.get("SEC_USER_AGENT", "MyApp myemail@email.com")

client = OpenAI(api_key=openai_key) if openai_key else None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Watchlist")
    tickers_input = st.text_input("Enter tickers (comma separated)", "NVDA, CRWD, PANW, AMD, SPY")
    period        = st.selectbox("Time period", ["3mo", "6mo", "1y", "2y"], index=1)
    selected      = st.selectbox(
        "Deep-dive ticker",
        [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    )

    st.divider()
    st.header("API Budget Today")

    av_used      = get_usage_today("alpha_vantage")
    av_remaining = requests_remaining("alpha_vantage", AV_DAILY_LIMIT)
    st.metric("Alpha Vantage", f"{av_remaining} / {AV_DAILY_LIMIT} left")
    st.progress(min(av_used / AV_DAILY_LIMIT, 1.0))
    if av_remaining <= 5:
        st.error(f"Only {av_remaining} AV requests left!")
    elif av_remaining <= 10:
        st.warning(f"{av_remaining} AV requests left today")

    fh_used      = get_usage_today("finnhub")
    fh_remaining = requests_remaining("finnhub", FINNHUB_DAILY_LIMIT)
    st.metric("Finnhub", f"{fh_remaining} / {FINNHUB_DAILY_LIMIT} left")
    st.progress(min(fh_used / FINNHUB_DAILY_LIMIT, 1.0))
    if fh_remaining <= 10:
        st.warning(f"{fh_remaining} Finnhub requests left today")

    st.caption("All counters reset at midnight.")

    st.divider()
    st.header("Phase 2 Status")
    st.caption("✅ News sentiment (Alpha Vantage)")
    st.caption("✅ Insider trades (Finnhub)")
    st.caption("✅ Phase 2 score adjustment")
    st.caption("✅ Phase 2 signal table")
    st.caption("✅ AI includes Phase 2 signals")
    st.caption("🔒 Politician trades (Phase 3)")
    st.caption("🔒 FDA catalysts (Phase 3)")
    st.caption("🔒 Earnings transcripts (Phase 3)")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ── Core helpers ──────────────────────────────────────────────────────────────
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

# ── Prompt 3: Score with optional Phase 2 signals ────────────────────────────
def score_stock(df, news_sentiment=None, insider_signal=None):
    score   = 50
    reasons = []
    latest  = df.iloc[-1]

    price   = safe_float(latest["Close"])
    ma20    = safe_float(latest["MA20"])
    ma50    = safe_float(latest["MA50"])
    rsi     = safe_float(latest["RSI"])
    vol     = safe_float(latest["Volume"])
    avg_vol = safe_float(latest["VolumeAvg"])

    # Phase 1 signals
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

    # Phase 2 signals
    if news_sentiment:
        label = news_sentiment.get("sentiment_label", "Neutral")
        avg   = news_sentiment.get("avg_score", 0)
        if label == "Bullish":
            score += 10
            reasons.append(f"News sentiment Bullish (score {avg:+.3f}) (+10)")
        elif label == "Bearish":
            score -= 10
            reasons.append(f"News sentiment Bearish (score {avg:+.3f}) (-10)")
        else:
            reasons.append(f"News sentiment Neutral (score {avg:+.3f}) (no change)")

    if insider_signal:
        if insider_signal == "Bullish":
            score += 10
            reasons.append("Insider activity Bullish — net buying (+10)")
        elif insider_signal == "Bearish":
            score -= 10
            reasons.append("Insider activity Bearish — net selling (-10)")
        else:
            reasons.append("Insider activity Neutral (no change)")

    return max(0, min(100, int(score))), reasons

def get_signal(score):
    if score >= 70:
        return "Bullish"
    elif score >= 50:
        return "Neutral"
    return "Bearish"

# ── Prompt 1: News sentiment ──────────────────────────────────────────────────
def fetch_news_sentiment(ticker):
    if not alpha_key:
        return None, "Alpha Vantage key missing"
    try:
        url  = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}&limit=20&apikey={alpha_key}"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if "Information" in data or "Note" in data:
            return None, data.get("Information") or data.get("Note")
        if "feed" not in data or not data["feed"]:
            return None, "No news found for this ticker"

        articles     = data["feed"]
        scores       = []
        top_articles = []

        for article in articles:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    try:
                        scores.append(float(ts["ticker_sentiment_score"]))
                    except:
                        pass
            if len(top_articles) < 3:
                top_articles.append({
                    "title":     article.get("title", "No title"),
                    "source":    article.get("source", "Unknown"),
                    "url":       article.get("url", ""),
                    "time":      article.get("time_published", "")[:8],
                    "sentiment": article.get("overall_sentiment_label", "Neutral")
                })

        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        label     = "Bullish" if avg_score >= 0.15 else ("Bearish" if avg_score <= -0.15 else "Neutral")

        return {
            "article_count":   len(articles),
            "avg_score":       avg_score,
            "sentiment_label": label,
            "top_articles":    top_articles,
            "scored_articles": len(scores)
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Prompt 2: Insider trades ──────────────────────────────────────────────────
def fetch_insider_transactions(ticker):
    if not finnhub_key:
        return None, "Finnhub key missing"
    try:
        url  = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={finnhub_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        transactions = data.get("data", [])
        if not transactions:
            return None, "No insider transactions found"

        cutoff = (datetime.now() - timedelta(days=90)).date()
        recent = []
        for t in transactions:
            try:
                tx_date = datetime.strptime(t.get("transactionDate", ""), "%Y-%m-%d").date()
                if tx_date >= cutoff:
                    recent.append(t)
            except:
                pass

        if not recent:
            return None, "No insider transactions in last 90 days"

        buys  = [t for t in recent if t.get("transactionCode") in ["P", "A"]]
        sells = [t for t in recent if t.get("transactionCode") in ["S", "D"]]

        net_shares = sum(t.get("share", 0) or 0 for t in buys) - sum(t.get("share", 0) or 0 for t in sells)

        if len(buys) > len(sells) and net_shares > 0:
            insider_signal = "Bullish"
        elif len(sells) > len(buys) and net_shares < 0:
            insider_signal = "Bearish"
        else:
            insider_signal = "Neutral"

        table_rows = []
        for t in recent[:10]:
            table_rows.append({
                "Date":        t.get("transactionDate", ""),
                "Name":        t.get("name", "Unknown"),
                "Type":        "BUY" if t.get("transactionCode") in ["P", "A"] else "SELL",
                "Shares":      f"{t.get('share', 0):,}",
                "Price":       f"${t.get('price', 0):.2f}" if t.get("price") else "N/A",
                "Value":       f"${(t.get('share', 0) or 0) * (t.get('price', 0) or 0):,.0f}"
            })

        return {
            "buy_count":       len(buys),
            "sell_count":      len(sells),
            "net_shares":      net_shares,
            "insider_signal":  insider_signal,
            "recent_count":    len(recent),
            "transactions":    table_rows
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Prompt 4: AI analysis with Phase 2 context ───────────────────────────────
def generate_ai_analysis(ticker, latest, score, sentiment_data=None, insider_data=None):
    sentiment_block = ""
    if sentiment_data:
        headlines = "\n".join([f"  - {a['title']} ({a['sentiment']})" for a in sentiment_data.get("top_articles", [])])
        sentiment_block = f"""
News Sentiment:
  Label: {sentiment_data['sentiment_label']}
  Avg Score: {sentiment_data['avg_score']}
  Articles Analyzed: {sentiment_data['article_count']}
  Top Headlines:
{headlines}
"""

    insider_block = ""
    if insider_data:
        insider_block = f"""
Insider Activity (last 90 days):
  Signal: {insider_data['insider_signal']}
  Buy transactions: {insider_data['buy_count']}
  Sell transactions: {insider_data['sell_count']}
  Net shares bought/sold: {insider_data['net_shares']:,}
"""

    prompt = f"""
You are an AI stock research assistant for paper trading only.
Analyze this ticker using technical indicators, news sentiment, and insider activity.

Ticker: {ticker}
Latest Close: ${safe_float(latest['Close']):.2f}
RSI: {safe_float(latest['RSI']):.1f}
Volume: {safe_float(latest['Volume']):.0f}
20-day MA: ${safe_float(latest['MA20']):.2f}
50-day MA: ${safe_float(latest['MA50']):.2f}
Combined Score: {score}/100
{sentiment_block}
{insider_block}

Please explain in plain English:
1. Whether the technical setup looks bullish, neutral, or weak and why
2. What the moving averages suggest about the trend
3. What the RSI level means right now
4. Whether volume supports or weakens the move
5. What the news sentiment suggests and how it supports or contradicts the technical picture (if available)
6. What insider buying or selling activity suggests (if available)
7. One specific thing a paper trader should watch for next

Keep it beginner-friendly. Reference the actual numbers.
This is for paper trading research only — not financial advice.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain stock setups clearly for beginner paper traders. Be specific, concise, and use plain English."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
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

def render_request_gate(api_name, limit, ticker, label, key_prefix):
    """Renders the request budget UI and confirmation button. Returns True if user confirmed."""
    used      = get_usage_today(api_name)
    remaining = requests_remaining(api_name, limit)

    c1, c2, c3 = st.columns(3)
    c1.metric("Remaining Today", f"{remaining} / {limit}")
    c2.metric("Used Today",      str(used))
    c3.metric("This call costs", "1 request")
    st.progress(min(used / limit, 1.0))

    if remaining <= 0:
        st.error(f"No {label} requests left today. Resets at midnight.")
        return False

    if remaining <= 5:
        st.error(f"Only {remaining} {label} requests left!")
    elif remaining <= 10:
        st.warning(f"{remaining} {label} requests left today")

    st.info(f"Fetching data for **{ticker}** costs **1 request** ({remaining - 1} remaining after).")
    return st.button(f"Confirm — fetch {label} data for {ticker}", key=f"{key_prefix}_confirm")

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
        latest   = data.iloc[-1]
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
                st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
            elif st.button(f"Generate AI Analysis for {ticker}", key=f"ai_{ticker}"):
                with st.spinner(f"Analyzing {ticker}..."):
                    analysis = generate_ai_analysis(ticker, latest, score)
                    st.write(analysis)

        st.divider()

# ── TAB 2: Deep Dive ──────────────────────────────────────────────────────────
with tab2:

    # ── Prompt 5: Phase 2 signal table ───────────────────────────────────────
    st.subheader("Phase 2 Signal Summary")
    st.caption("Uses only locally cached Phase 2 data — no new API calls made here.")

    p2_rows = []
    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty:
            continue
        data = add_indicators(raw.copy())
        if data.empty:
            continue

        cached_sentiment = st.session_state.get(f"sentiment_{ticker}")
        cached_insider   = st.session_state.get(f"insider_{ticker}")

        sent_label     = cached_sentiment["sentiment_label"] if cached_sentiment else "—"
        insider_signal = cached_insider["insider_signal"]    if cached_insider   else "—"

        tech_score, _ = score_stock(data)
        combined_score, _ = score_stock(
            data,
            news_sentiment = cached_sentiment,
            insider_signal = insider_signal if insider_signal != "—" else None
        )
        final_signal = get_signal(combined_score)

        p2_rows.append({
            "Ticker":         ticker,
            "Tech Score":     f"{tech_score}/100",
            "News Sentiment": sent_label,
            "Insider Signal": insider_signal,
            "Combined Score": f"{combined_score}/100",
            "Final Signal":   final_signal
        })

    if p2_rows:
        p2_df = pd.DataFrame(p2_rows)
        p2_df["_sort"] = p2_df["Combined Score"].str.replace("/100", "").astype(int)
        p2_df = p2_df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        st.dataframe(p2_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Main deep dive ────────────────────────────────────────────────────────
    st.subheader(f"Deep Dive: {selected}")
    raw = get_data(selected, period)

    if raw.empty:
        st.error(f"No data for {selected}")
    else:
        data   = add_indicators(raw.copy())
        latest = data.iloc[-1]

        cached_sentiment = st.session_state.get(f"sentiment_{selected}")
        cached_insider   = st.session_state.get(f"insider_{selected}")
        insider_sig      = cached_insider["insider_signal"] if cached_insider else None

        score, reasons = score_stock(
            data,
            news_sentiment = cached_sentiment,
            insider_signal = insider_sig
        )
        signal = get_signal(score)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Combined Score", f"{score}/100")
        col2.metric("Signal",         signal)
        col3.metric("RSI",            f"{safe_float(latest['RSI']):.1f}")
        col4.metric("Close",          f"${safe_float(latest['Close']):.2f}")

        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price"))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(color="orange")))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(color="royalblue")))
        fig2.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected} — Deep Dive")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Score breakdown"):
            for r in reasons:
                st.write(f"• {r}")

        # ── News sentiment expander ───────────────────────────────────────────
        with st.expander("News Sentiment (Alpha Vantage)", expanded=False):
            if not alpha_key:
                st.warning("Add ALPHA_VANTAGE_API_KEY to Streamlit Secrets.")
            else:
                confirmed = render_request_gate(
                    "alpha_vantage", AV_DAILY_LIMIT, selected,
                    "Alpha Vantage", f"av_{selected}"
                )
                if confirmed:
                    with st.spinner(f"Fetching news sentiment for {selected}..."):
                        result, err = fetch_news_sentiment(selected)
                    if err:
                        st.error(f"Could not load sentiment: {err}")
                    else:
                        increment_usage("alpha_vantage", 1)
                        st.session_state[f"sentiment_{selected}"] = result
                        st.rerun()

                cached = st.session_state.get(f"sentiment_{selected}")
                if cached:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Sentiment",      cached["sentiment_label"])
                    s2.metric("Avg Score",      cached["avg_score"])
                    s3.metric("Articles Found", cached["article_count"])

                    if cached["sentiment_label"] == "Bullish":
                        st.success("News sentiment is Bullish")
                    elif cached["sentiment_label"] == "Bearish":
                        st.error("News sentiment is Bearish")
                    else:
                        st.info("News sentiment is Neutral")

                    st.markdown("**Top recent headlines:**")
                    for article in cached["top_articles"]:
                        t = article["time"]
                        if len(t) == 8:
                            t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
                        st.markdown(f"- [{article['title']}]({article['url']})  \n  *{article['source']} · {t} · {article['sentiment']}*")
                else:
                    st.caption("No sentiment data loaded yet. Confirm above to fetch.")

        # ── Insider trades expander ───────────────────────────────────────────
        with st.expander("Insider Activity (Finnhub)", expanded=False):
            if not finnhub_key:
                st.warning("Add FINNHUB_API_KEY to Streamlit Secrets.")
            else:
                confirmed_fh = render_request_gate(
                    "finnhub", FINNHUB_DAILY_LIMIT, selected,
                    "Finnhub", f"fh_{selected}"
                )
                if confirmed_fh:
                    with st.spinner(f"Fetching insider transactions for {selected}..."):
                        insider_result, insider_err = fetch_insider_transactions(selected)
                    if insider_err:
                        st.error(f"Could not load insider data: {insider_err}")
                    else:
                        increment_usage("finnhub", 1)
                        st.session_state[f"insider_{selected}"] = insider_result
                        st.rerun()

                cached_ins = st.session_state.get(f"insider_{selected}")
                if cached_ins:
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Insider Signal", cached_ins["insider_signal"])
                    i2.metric("Buys (90d)",     cached_ins["buy_count"])
                    i3.metric("Sells (90d)",     cached_ins["sell_count"])

                    net = cached_ins["net_shares"]
                    if cached_ins["insider_signal"] == "Bullish":
                        st.success(f"Net insider buying: {net:,} shares")
                    elif cached_ins["insider_signal"] == "Bearish":
                        st.error(f"Net insider selling: {net:,} shares")
                    else:
                        st.info(f"Mixed insider activity. Net shares: {net:,}")

                    if cached_ins["transactions"]:
                        st.markdown("**Recent insider transactions:**")
                        st.dataframe(
                            pd.DataFrame(cached_ins["transactions"]),
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.caption("No insider data loaded yet. Confirm above to fetch.")

        # ── AI summary ────────────────────────────────────────────────────────
        st.subheader("AI Summary")
        if client is None:
            st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
        elif st.button("Generate AI Summary", key="ai_deepdive"):
            with st.spinner("Analyzing..."):
                analysis = generate_ai_analysis(
                    selected, latest, score,
                    sentiment_data = st.session_state.get(f"sentiment_{selected}"),
                    insider_data   = st.session_state.get(f"insider_{selected}")
                )
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
        trade_price   = c3.number_input("Entry Price", value=default_price)

        raw_ind = get_data(trade_ticker, period)
        if not raw_ind.empty:
            ind_data       = add_indicators(raw_ind.copy())
            ind_latest     = ind_data.iloc[-1]
            trade_score, _ = score_stock(ind_data)
            trade_signal   = get_signal(trade_score)
            trade_rsi      = safe_float(ind_latest["RSI"])
            trade_ma20     = safe_float(ind_latest["MA20"])
            trade_ma50     = safe_float(ind_latest["MA50"])
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
        st.info("No trades logged yet.")
    else:
        st.subheader("Open Trades")
        open_trades = trades_df[trades_df["status"] == "Open"].copy()

        if not open_trades.empty:
            perf_rows = []
            for _, row in open_trades.iterrows():
                current_price = get_current_price(row["ticker"])
                if current_price and row["entry_price"] > 0:
                    gain_pct = ((current_price - row["entry_price"]) / row["entry_price"]) * 100
                    if row["action"] == "SELL":
                        gain_pct = -gain_pct
                    gain_str = f"{gain_pct:+.2f}%"
                else:
                    current_price = 0.0
                    gain_str      = "N/A"

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
