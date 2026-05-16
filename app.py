import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("AI Trading Research Dashboard")
st.caption("Phase 1: Watchlist, charts, indicators, scoring, and AI summaries")

# ── API key validation ────────────────────────────────────────────────────────
openai_key = st.secrets.get("OPENAI_API_KEY", None)
alpha_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)

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
    selected = st.selectbox("Deep-dive ticker", [t.strip().upper() for t in tickers_input.split(",") if t.strip()])

    st.divider()
    st.header("Phase 2 (coming soon)")
    st.caption("🔒 SEC insider trades")
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
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df)
    df["VolumeAvg"] = df["Volume"].rolling(20).mean()
    return df.dropna()

def score_stock(df):
    score = 50
    reasons = []
    latest = df.iloc[-1]

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

def generate_ai_analysis(ticker, latest, score):
    prompt = f"""
    You are an AI stock research assistant for paper trading only.
    Analyze this ticker using the provided technical indicators.

    Ticker: {ticker}
    Latest Close: {safe_float(latest['Close']):.2f}
    RSI: {safe_float(latest['RSI']):.1f}
    Volume: {safe_float(latest['Volume']):.0f}
    20-day Moving Average: {safe_float(latest['MA20']):.2f}
    50-day Moving Average: {safe_float(latest['MA50']):.2f}
    Score: {score}/100

    Please explain:
    1. Whether this looks bullish, neutral, or weak and why
    2. What the moving averages suggest about the trend
    3. What the RSI level means right now
    4. Whether volume supports or weakens the move
    5. One specific thing a paper trader should watch for next

    Keep it beginner-friendly. Be specific to this ticker's numbers.
    End with a disclaimer that this is not financial advice.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain stock setups clearly for beginner paper traders. Be specific, concise, and use plain English."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content

@st.cache_data(ttl=300)
def get_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Watchlist Overview", "Deep Dive", "Paper Trade Log"])

# ── TAB 1: Watchlist overview ─────────────────────────────────────────────────
with tab1:
    for ticker in tickers:
        st.subheader(ticker)
        raw = get_data(ticker, period)

        if raw.empty:
            st.error(f"No data found for {ticker}")
            continue

        data = add_indicators(raw.copy())

        if data.empty:
            st.warning(f"Not enough data yet for {ticker}")
            continue

        latest = data.iloc[-1]
        score, reasons = score_stock(data)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close",    f"${safe_float(latest['Close']):.2f}")
        col2.metric("RSI",      f"{safe_float(latest['RSI']):.1f}")
        col3.metric("Volume",   f"{safe_float(latest['Volume']):,.0f}")
        col4.metric("AI Score", f"{score}/100")

        if score >= 70:
            st.success("Bullish — strong setup")
        elif score >= 50:
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

        # AI Analysis button per ticker
        with st.expander("AI Assistant Analysis"):
            if client is None:
                st.warning("Add your OPENAI_API_KEY in Streamlit Secrets to enable this.")
            else:
                if st.button(f"Generate AI Analysis for {ticker}", key=f"ai_{ticker}"):
                    with st.spinner(f"Analyzing {ticker}..."):
                        analysis = generate_ai_analysis(ticker, latest, score)
                        st.write(analysis)

        st.divider()

# ── TAB 2: Deep dive ──────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Deep Dive: {selected}")
    raw = get_data(selected, period)

    if raw.empty:
        st.error(f"No data for {selected}")
    else:
        data = add_indicators(raw.copy())
        latest = data.iloc[-1]
        score, reasons = score_stock(data)

        if score >= 70:
            sentiment = "Bullish"
        elif score >= 50:
            sentiment = "Neutral"
        else:
            sentiment = "Bearish"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score",  f"{score}/100")
        col2.metric("Signal", sentiment)
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

        st.subheader("AI Summary")
        if client is None:
            st.warning("Add your OPENAI_API_KEY in Streamlit Secrets to enable this.")
        elif st.button("Generate AI Summary", key="ai_deepdive"):
            with st.spinner("Analyzing..."):
                analysis = generate_ai_analysis(selected, latest, score)
                st.info(analysis)

# ── TAB 3: Paper trade log ────────────────────────────────────────────────────
with tab3:
    st.subheader("Paper Trade Log")
    st.caption("Track your hypothetical trades here. No real money involved.")

    if "trades" not in st.session_state:
        st.session_state.trades = []

    with st.expander("Log a new trade"):
        c1, c2, c3 = st.columns(3)
        trade_ticker = c1.text_input("Ticker", value=selected)
        trade_action = c2.selectbox("Action", ["BUY", "SELL"])

        raw_log = get_data(trade_ticker, "5d")
        if isinstance(raw_log.columns, pd.MultiIndex):
            raw_log.columns = raw_log.columns.get_level_values(0)
        default_price = float(raw_log["Close"].iloc[-1]) if not raw_log.empty else 100.0
        trade_price = c3.number_input("Price", value=default_price)

        trade_notes = st.text_input("Notes (optional)")
        if st.button("Log Trade"):
            st.session_state.trades.append({
                "Ticker": trade_ticker,
                "Action": trade_action,
                "Price": f"${trade_price:.2f}",
                "Notes": trade_notes
            })
            st.success(f"Logged {trade_action} {trade_ticker} at ${trade_price:.2f}")

    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True)
        if st.button("Clear all trades"):
            st.session_state.trades = []
            st.rerun()
    else:
        st.info("No trades logged yet. Use the form above to add one.")
