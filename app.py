import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from datetime import datetime, date, timedelta
import sqlite3
import os
import requests
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Research Dashboard", layout="wide")
st.title("AI Trading Research Dashboard")
st.caption(
    "📊 An educational paper-trading research tool. "
    "All signals, scores, and AI summaries are for research and educational purposes only. "
    "Nothing here is financial advice. Past patterns do not guarantee future results."
)
st.info(
    "⚠️ **Disclaimer:** This dashboard is for personal research and paper-trading education only. "
    "It does not provide financial advice, investment recommendations, or trading signals. "
    "All data shown may be delayed, incomplete, or inaccurate. "
    "Never make real financial decisions based solely on this tool. "
    "Consult a licensed financial professional before investing."
)

# ── Database setup ────────────────────────────────────────────────────────────
DB_PATH = "data/trades.db"
os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT, ticker TEXT, action TEXT,
            entry_price REAL, ai_score INTEGER, rsi REAL,
            ma20 REAL, ma50 REAL, signal TEXT, notes TEXT,
            status TEXT DEFAULT 'Open'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_name TEXT, usage_date TEXT, count INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            ticker TEXT, fetched_date TEXT, article_count INTEGER,
            avg_score REAL, sentiment_label TEXT, top_articles TEXT,
            PRIMARY KEY (ticker, fetched_date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS insider_cache (
            ticker TEXT, fetched_date TEXT, buy_count INTEGER,
            sell_count INTEGER, net_shares REAL, insider_signal TEXT,
            transactions TEXT, PRIMARY KEY (ticker, fetched_date)
        )
    """)
    # Phase 3 — Prompt 1: signal history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            snapshot_date TEXT,
            close REAL,
            rsi REAL,
            ma20 REAL,
            ma50 REAL,
            volume REAL,
            technical_score INTEGER,
            news_sentiment_label TEXT,
            insider_signal TEXT,
            combined_score INTEGER,
            final_signal TEXT,
            notes TEXT,
            UNIQUE(ticker, snapshot_date)
        )
    """)
    conn.commit()
    conn.close()

# ── Trade CRUD ────────────────────────────────────────────────────────────────
def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()
    return df

def save_trade(entry_date, ticker, action, entry_price, ai_score, rsi, ma20, ma50, signal, notes):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO trades
        (entry_date,ticker,action,entry_price,ai_score,rsi,ma20,ma50,signal,notes,status)
        VALUES (?,?,?,?,?,?,?,?,?,?,'Open')
    """, (entry_date, ticker, action, entry_price, ai_score, rsi, ma20, ma50, signal, notes))
    conn.commit()
    conn.close()

def close_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("UPDATE trades SET status='Closed' WHERE id=?", (trade_id,))
    conn.commit(); conn.close()

def delete_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM trades WHERE id=?", (trade_id,))
    conn.commit(); conn.close()

# ── News/Insider cache ────────────────────────────────────────────────────────
def save_news_cache(ticker, result):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        INSERT OR REPLACE INTO news_cache
        (ticker,fetched_date,article_count,avg_score,sentiment_label,top_articles)
        VALUES (?,?,?,?,?,?)
    """, (ticker, today, result["article_count"], result["avg_score"],
          result["sentiment_label"], json.dumps(result["top_articles"])))
    conn.commit(); conn.close()

def load_news_cache(ticker):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("""
        SELECT article_count,avg_score,sentiment_label,top_articles
        FROM news_cache WHERE ticker=? AND fetched_date=?
    """, (ticker, today))
    row = c.fetchone(); conn.close()
    if row:
        return {"article_count": row[0], "avg_score": row[1],
                "sentiment_label": row[2], "top_articles": json.loads(row[3])}
    return None

def save_insider_cache(ticker, result):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        INSERT OR REPLACE INTO insider_cache
        (ticker,fetched_date,buy_count,sell_count,net_shares,insider_signal,transactions)
        VALUES (?,?,?,?,?,?,?)
    """, (ticker, today, result["buy_count"], result["sell_count"],
          result["net_shares"], result["insider_signal"],
          json.dumps(result["transactions"])))
    conn.commit(); conn.close()

def load_insider_cache(ticker):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("""
        SELECT buy_count,sell_count,net_shares,insider_signal,transactions
        FROM insider_cache WHERE ticker=? AND fetched_date=?
    """, (ticker, today))
    row = c.fetchone(); conn.close()
    if row:
        return {"buy_count": row[0], "sell_count": row[1], "net_shares": row[2],
                "insider_signal": row[3], "transactions": json.loads(row[4])}
    return None

# ── Phase 3: Signal history ───────────────────────────────────────────────────
def save_signal_snapshot(ticker, close, rsi, ma20, ma50, volume,
                          technical_score, news_sentiment_label,
                          insider_signal, combined_score, final_signal, notes=""):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    try:
        c.execute("""
            INSERT INTO signal_history
            (ticker,snapshot_date,close,rsi,ma20,ma50,volume,
             technical_score,news_sentiment_label,insider_signal,
             combined_score,final_signal,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ticker, today, close, rsi, ma20, ma50, volume,
              technical_score, news_sentiment_label or "Unavailable",
              insider_signal or "Unavailable",
              combined_score, final_signal, notes))
        conn.commit()
        result = "saved"
    except sqlite3.IntegrityError:
        result = "exists"
    conn.close()
    return result

def load_signal_history():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM signal_history ORDER BY snapshot_date DESC", conn)
    conn.close()
    return df

init_db()

# ── API counter ───────────────────────────────────────────────────────────────
AV_DAILY_LIMIT      = 25
FINNHUB_DAILY_LIMIT = 60
OPENAI_DAILY_LIMIT  = 20

def get_usage_today(api_name):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT count FROM api_usage WHERE api_name=? AND usage_date=?", (api_name, today))
    row = c.fetchone(); conn.close()
    return row[0] if row else 0

def increment_usage(api_name, amount=1):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT count FROM api_usage WHERE api_name=? AND usage_date=?", (api_name, today))
    row = c.fetchone()
    if row:
        c.execute("UPDATE api_usage SET count=count+? WHERE api_name=? AND usage_date=?",
                  (amount, api_name, today))
    else:
        c.execute("INSERT INTO api_usage (api_name,usage_date,count) VALUES (?,?,?)",
                  (api_name, today, amount))
    conn.commit(); conn.close()

def requests_remaining(api_name, limit):
    return max(0, limit - get_usage_today(api_name))

# ── API keys ──────────────────────────────────────────────────────────────────
openai_key  = st.secrets.get("OPENAI_API_KEY", None)
alpha_key   = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
finnhub_key = st.secrets.get("FINNHUB_API_KEY", None)
sec_agent   = st.secrets.get("SEC_USER_AGENT", "MyApp myemail@email.com")
client      = OpenAI(api_key=openai_key) if openai_key else None

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
    for api, label, limit in [
        ("alpha_vantage", "Alpha Vantage", AV_DAILY_LIMIT),
        ("finnhub",       "Finnhub",       FINNHUB_DAILY_LIMIT),
        ("openai",        "OpenAI calls",  OPENAI_DAILY_LIMIT),
    ]:
        used = get_usage_today(api)
        rem  = requests_remaining(api, limit)
        st.metric(label, f"{rem} / {limit} left")
        st.progress(min(used / limit, 1.0))
        if rem <= 5:
            st.error(f"Only {rem} {label} requests left!")
        elif rem <= 10:
            st.warning(f"{rem} {label} requests left today")
    st.caption("All counters reset at midnight.")

    st.divider()
    st.header("Build Status")
    st.caption("✅ Phase 1 — Watchlist + charts + scoring")
    st.caption("✅ Phase 2 — News sentiment + insider trades")
    st.caption("✅ Phase 3 — Signal history + backtesting + scanner")
    st.caption("🔒 Phase 4 — Public disclosure tracker")
    st.caption("🔒 Phase 4 — FDA catalysts")
    st.caption("🔒 Phase 4 — Earnings transcripts")

    st.divider()
    st.caption(
        "📋 For educational and paper-trading research only. "
        "Not financial advice. Consult a licensed financial professional "
        "before making any investment decisions."
    )

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ── Disclaimer constant ───────────────────────────────────────────────────────
DISCLAIMER = (
    "*For educational and paper-trading research only. "
    "Not financial advice. Signals may be delayed or inaccurate. "
    "Never make real investment decisions based solely on this tool.*"
)

# ── Discovery watchlists ──────────────────────────────────────────────────────
DISCOVERY_LISTS = {
    "AI & Machine Learning":    ["NVDA", "AMD", "MSFT", "GOOGL", "META", "AMZN", "ORCL", "IBM", "PLTR", "AI"],
    "Cybersecurity":            ["CRWD", "PANW", "FTNT", "ZS", "S",    "OKTA", "CYBR", "TENB", "RPD", "SAIL"],
    "Semiconductors":           ["NVDA", "AMD",  "INTC", "QCOM", "AVGO", "MU",   "AMAT", "LRCX", "KLAC", "TSM"],
    "Defense":                  ["LMT",  "RTX",  "NOC",  "GD",   "BA",   "L3HT", "HII",  "LDOS", "CACI", "SAIC"],
    "Biotech":                  ["MRNA", "BNTX", "REGN", "VRTX", "BIIB", "GILD", "AMGN", "ILMN", "RARE", "EXAS"],
    "Cloud & SaaS":             ["CRM",  "NOW",  "SNOW", "DDOG", "MDB",  "NET",  "HUBS", "ZM",   "TEAM", "WDAY"],
    "Major ETFs":               ["SPY",  "QQQ",  "IWM",  "DIA",  "XLK",  "XLF",  "XLV",  "GLD",  "TLT",  "VIX"],
    "Mega-Cap Tech":            ["AAPL", "MSFT", "GOOGL","AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
}

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

    if price > ma50:
        score += 20; reasons.append("Price above 50-day MA — historically favorable (+20)")
    else:
        score -= 20; reasons.append("Price below 50-day MA — historically weaker setup (-20)")

    if ma20 > ma50:
        score += 15; reasons.append("20-day MA above 50-day MA — bullish alignment signal (+15)")

    if 45 <= rsi <= 70:
        score += 15; reasons.append(f"RSI {rsi:.1f} — momentum in healthy range (+15)")
    elif rsi > 75:
        score -= 15; reasons.append(f"RSI {rsi:.1f} — potentially overbought (-15)")
    else:
        reasons.append(f"RSI {rsi:.1f} — outside ideal range, watch for direction (no change)")

    if vol > avg_vol:
        score += 10; reasons.append("Volume above 20-day average — move may have conviction (+10)")
    else:
        reasons.append("Volume below average — move may lack conviction (no change)")

    if news_sentiment:
        label = news_sentiment.get("sentiment_label", "Neutral")
        avg   = news_sentiment.get("avg_score", 0)
        if label == "Bullish":
            score += 10; reasons.append(f"News sentiment leaning bullish ({avg:+.3f}) (+10)")
        elif label == "Bearish":
            score -= 10; reasons.append(f"News sentiment leaning bearish ({avg:+.3f}) (-10)")
        else:
            reasons.append(f"News sentiment neutral ({avg:+.3f}) (no change)")

    if insider_signal:
        if insider_signal == "Bullish":
            score += 10; reasons.append("Open-market insider buying detected (+10)")
        elif insider_signal == "Bearish":
            score -= 5;  reasons.append("Insider selling detected — may be routine (-5)")
        else:
            reasons.append("Insider activity mixed or neutral (no change)")

    return max(0, min(100, int(score))), reasons

def get_signal(score):
    if score >= 70: return "Bullish signals align"
    if score >= 50: return "Neutral — watch carefully"
    return "Weak setup — exercise caution"

def get_signal_short(score):
    if score >= 70: return "Bullish"
    if score >= 50: return "Neutral"
    return "Weak"

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

# ── News sentiment ────────────────────────────────────────────────────────────
def fetch_news_sentiment(ticker):
    if not alpha_key:
        return None, "Alpha Vantage key missing"
    try:
        url = (f"https://www.alphavantage.co/query"
               f"?function=NEWS_SENTIMENT&tickers={ticker}&limit=20&apikey={alpha_key}")
        increment_usage("alpha_vantage", 1)
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "Information" in data or "Note" in data:
            return None, data.get("Information") or data.get("Note")
        if "feed" not in data or not data["feed"]:
            return None, "No news found"
        articles = data["feed"]
        scores, top_articles = [], []
        for article in articles:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    try: scores.append(float(ts["ticker_sentiment_score"]))
                    except: pass
            if len(top_articles) < 5:
                top_articles.append({
                    "title":     article.get("title", "No title"),
                    "source":    article.get("source", "Unknown"),
                    "url":       article.get("url", ""),
                    "time":      article.get("time_published", "")[:8],
                    "sentiment": article.get("overall_sentiment_label", "Neutral"),
                    "summary":   article.get("summary", "")
                })
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        label = "Bullish" if avg_score >= 0.15 else ("Bearish" if avg_score <= -0.15 else "Neutral")
        return {"article_count": len(articles), "avg_score": avg_score,
                "sentiment_label": label, "top_articles": top_articles,
                "scored_articles": len(scores)}, None
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Insider trades ────────────────────────────────────────────────────────────
def fetch_insider_transactions(ticker):
    if not finnhub_key:
        return None, "Finnhub key missing"
    try:
        url = (f"https://finnhub.io/api/v1/stock/insider-transactions"
               f"?symbol={ticker}&token={finnhub_key}")
        increment_usage("finnhub", 1)
        resp = requests.get(url, timeout=10)
        data = resp.json()
        transactions = data.get("data", [])
        if not transactions:
            return None, "No insider transactions found"
        cutoff = (datetime.now() - timedelta(days=90)).date()
        recent = []
        for t in transactions:
            try:
                if datetime.strptime(t.get("transactionDate", ""), "%Y-%m-%d").date() >= cutoff:
                    recent.append(t)
            except: pass
        if not recent:
            return None, "No transactions in last 90 days"
        open_buys  = [t for t in recent if t.get("transactionCode") == "P"]
        all_buys   = [t for t in recent if t.get("transactionCode") in ["P", "A"]]
        open_sells = [t for t in recent if t.get("transactionCode") == "S"]
        all_sells  = [t for t in recent if t.get("transactionCode") in ["S", "D"]]
        net_shares = (sum(t.get("share", 0) or 0 for t in all_buys) -
                      sum(t.get("share", 0) or 0 for t in all_sells))
        if len(open_buys) > 0 and len(open_buys) >= len(open_sells):
            insider_signal = "Bullish"
        elif len(open_sells) > len(open_buys) * 2:
            insider_signal = "Bearish"
        else:
            insider_signal = "Neutral"
        table_rows = []
        for t in recent[:10]:
            code = t.get("transactionCode", "")
            table_rows.append({
                "Date":   t.get("transactionDate", ""),
                "Name":   t.get("name", "Unknown"),
                "Type":   ("Open-Market Buy"   if code == "P" else
                           "Grant/Award"        if code == "A" else
                           "Open-Market Sell"   if code == "S" else
                           "Planned/Auto Sell"),
                "Shares": f"{t.get('share', 0):,}",
                "Price":  f"${t.get('price', 0):.2f}" if t.get("price") else "N/A",
                "Value":  f"${(t.get('share', 0) or 0) * (t.get('price', 0) or 0):,.0f}"
            })
        return {"buy_count": len(all_buys), "sell_count": len(all_sells),
                "open_buy_count": len(open_buys), "open_sell_count": len(open_sells),
                "net_shares": net_shares, "insider_signal": insider_signal,
                "recent_count": len(recent), "transactions": table_rows}, None
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── ChatGPT prompt builder ────────────────────────────────────────────────────
def build_chatgpt_prompt(ticker, latest, score, reasons, sentiment_data=None, insider_data=None):
    lines = [
        "You are an AI stock research assistant for paper trading and educational purposes only.",
        f"Analyze the following publicly available data for {ticker}.",
        "",
        "Return clearly labeled sections:",
        "1. Bullish factors (what looks favorable)",
        "2. Bearish risks (what looks concerning)",
        "3. Neutral or unclear points",
        "4. Possible near-term market impact to watch",
        "5. What a paper trader might monitor next",
        "",
        "Rules: Do not say buy or sell. Do not predict prices. Use hedged language.",
        "End with: This analysis is for paper trading and educational research only. Not financial advice.",
        "",
        f"--- TECHNICAL DATA FOR {ticker} ---",
        f"Close: ${safe_float(latest['Close']):.2f}",
        f"RSI: {safe_float(latest['RSI']):.1f}",
        f"MA20: ${safe_float(latest['MA20']):.2f}",
        f"MA50: ${safe_float(latest['MA50']):.2f}",
        f"Volume: {safe_float(latest['Volume']):,.0f}",
        f"Research Score: {score}/100 (educational metric only)",
        "", "Score signals:",
    ]
    for r in reasons:
        lines.append(f"  - {r}")
    if sentiment_data:
        lines += ["", "--- PUBLIC NEWS SENTIMENT ---",
                  f"Label: {sentiment_data['sentiment_label']}",
                  f"Avg Score: {sentiment_data['avg_score']}",
                  f"Articles: {sentiment_data['article_count']}", "Headlines:"]
        for a in sentiment_data["top_articles"]:
            lines.append(f"  - [{a['sentiment']}] {a['title']} ({a['source']})")
            if a.get("summary"):
                lines.append(f"    {a['summary'][:200]}...")
    if insider_data:
        lines += ["", "--- PUBLIC INSIDER DISCLOSURES (SEC filings, last 90 days) ---",
                  f"Signal: {insider_data['insider_signal']}",
                  f"Open-market buys: {insider_data.get('open_buy_count', insider_data['buy_count'])}",
                  f"Open-market sells: {insider_data.get('open_sell_count', insider_data['sell_count'])}",
                  f"Net shares: {insider_data['net_shares']:,}", "Recent:"]
        for t in insider_data["transactions"][:5]:
            lines.append(f"  - {t['Date']} | {t['Name']} | {t['Type']} | {t['Shares']} @ {t['Price']}")
    lines += ["", "---",
              "Reminder: For paper trading and educational research only. Not financial advice."]
    return "\n".join(lines)

# ── OpenAI helpers ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI research assistant helping with paper trading education. "
    "Never say buy or sell. Never predict prices. Use hedged language always: "
    "'signals suggest', 'may indicate', 'historically'. "
    "End every response with: This analysis is for paper trading and educational research only. "
    "Not financial advice."
)

def generate_ai_analysis(ticker, latest, score, sentiment_data=None, insider_data=None):
    s_block = ""
    if sentiment_data:
        headlines = "\n".join([
            f"  - [{a['sentiment']}] {a['title']} — {a.get('summary','')[:150]}"
            for a in sentiment_data.get("top_articles", [])
        ])
        s_block = f"\nNews Sentiment: {sentiment_data['sentiment_label']} (score {sentiment_data['avg_score']})\nHeadlines:\n{headlines}\n"
    i_block = ""
    if insider_data:
        i_block = (f"\nPublic Insider Disclosures (90 days): {insider_data['insider_signal']}\n"
                   f"Open-mkt buys: {insider_data.get('open_buy_count', insider_data['buy_count'])} | "
                   f"Open-mkt sells: {insider_data.get('open_sell_count', insider_data['sell_count'])}\n"
                   f"Net shares: {insider_data['net_shares']:,}\n")
    prompt = f"""Analyze for paper trading research: {ticker}
Close: ${safe_float(latest['Close']):.2f} | RSI: {safe_float(latest['RSI']):.1f}
MA20: ${safe_float(latest['MA20']):.2f} | MA50: ${safe_float(latest['MA50']):.2f}
Volume: {safe_float(latest['Volume']):.0f} | Research Score: {score}/100
{s_block}{i_block}
Return: 1) Technical setup assessment 2) MA trend 3) RSI reading
4) Volume conviction 5) News signals (if available) 6) Insider signals (if available)
7) One thing to watch next. Hedged language throughout."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=0.3, max_tokens=600
    )
    return resp.choices[0].message.content

def analyze_articles_with_ai(ticker, articles):
    article_text = "\n".join([
        f"Title: {a['title']}\nSource: {a['source']} | {a['sentiment']}\n"
        f"Summary: {a.get('summary','')[:300]}\nURL: {a['url']}"
        for a in articles
    ])
    prompt = f"""Analyze these public news articles for {ticker} (paper trading research only).
{article_text}
Return: 1) Bullish factors 2) Bearish risks 3) Neutral points
4) Possible market impact 5) What to watch next. No buy/sell. Hedged language."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": prompt}],
        temperature=0.3, max_tokens=600
    )
    return resp.choices[0].message.content

# ── UI helpers ────────────────────────────────────────────────────────────────
def render_request_gate(api_name, limit, ticker, label, key_prefix):
    used = get_usage_today(api_name)
    rem  = requests_remaining(api_name, limit)
    c1, c2, c3 = st.columns(3)
    c1.metric("Remaining", f"{rem}/{limit}")
    c2.metric("Used Today", str(used))
    c3.metric("This call", "1 request")
    st.progress(min(used / limit, 1.0))
    if rem <= 0:
        st.error(f"No {label} requests left today. Resets at midnight.")
        return False
    if rem <= 5:  st.error(f"Only {rem} {label} requests left!")
    elif rem <= 10: st.warning(f"{rem} {label} requests left today")
    st.info(f"Fetching **{label}** for **{ticker}** costs 1 request ({rem-1} after).")
    return st.button(f"Confirm — fetch {label} for {ticker}", key=f"{key_prefix}_confirm")

def render_openai_gate(key_prefix, label):
    used = get_usage_today("openai")
    rem  = requests_remaining("openai", OPENAI_DAILY_LIMIT)
    st.caption(f"OpenAI: {used} used / {rem} remaining today (limit: {OPENAI_DAILY_LIMIT})")
    if rem <= 0:
        st.error("Hit your self-set OpenAI limit for today.")
        return False
    if rem <= 3: st.error(f"Only {rem} OpenAI calls left!")
    return st.button(f"Use OpenAI — {label} (1 call + tokens)", key=key_prefix)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Watchlist Overview",
    "Deep Dive + Phase 2",
    "Backtesting",
    "Stock Discovery",
    "Paper Trade Log"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Watchlist Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Watchlist Summary")
    st.caption("Research scores are educational metrics only. Higher score = more bullish technical alignment. Not a trade recommendation.")
    summary_rows = []
    all_data     = {}

    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty: continue
        data = add_indicators(raw.copy())
        if data.empty: continue
        all_data[ticker] = data
        latest   = data.iloc[-1]
        score, _ = score_stock(data)
        summary_rows.append({
            "Ticker": ticker,
            "Close":  f"${safe_float(latest['Close']):.2f}",
            "RSI":    f"{safe_float(latest['RSI']):.1f}",
            "MA20":   f"${safe_float(latest['MA20']):.2f}",
            "MA50":   f"${safe_float(latest['MA50']):.2f}",
            "Research Score": f"{score}/100",
            "Signal Alignment": get_signal_short(score)
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf["_s"] = sdf["Research Score"].str.replace("/100","").astype(int)
        sdf = sdf.sort_values("_s", ascending=False).drop(columns=["_s"])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
    st.caption(DISCLAIMER)
    st.divider()

    for ticker in tickers:
        if ticker not in all_data:
            st.error(f"No data for {ticker}"); continue
        data   = all_data[ticker]
        latest = data.iloc[-1]
        score, reasons = score_stock(data)
        signal = get_signal(score)

        st.subheader(ticker)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close",          f"${safe_float(latest['Close']):.2f}")
        c2.metric("RSI",            f"{safe_float(latest['RSI']):.1f}")
        c3.metric("Volume",         f"{safe_float(latest['Volume']):,.0f}")
        c4.metric("Research Score", f"{score}/100")

        if score >= 70:   st.success(f"Signal alignment: {signal}")
        elif score >= 50: st.info(f"Signal alignment: {signal}")
        else:             st.warning(f"Signal alignment: {signal}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                     low=data["Low"], close=data["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(color="royalblue")))
        fig.update_layout(height=400, xaxis_rangeslider_visible=False,
                          title=f"{ticker} — {period} (data may be delayed ~15 min)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Score breakdown (research signals only)"):
            st.caption("Educational indicators only. Do not predict future performance.")
            for r in reasons: st.write(f"• {r}")
            st.caption(DISCLAIMER)

        with st.expander("AI Research Summary (uses OpenAI API)"):
            if client is None:
                st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
            else:
                st.caption("AI summaries are for research only. AI is instructed not to recommend trades.")
                if render_openai_gate(f"ai_{ticker}", f"Generate summary for {ticker}"):
                    increment_usage("openai", 1)
                    with st.spinner(f"Generating summary for {ticker}..."):
                        st.write(generate_ai_analysis(ticker, latest, score))
                    st.caption(DISCLAIMER)
        st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Dive + Phase 2
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Phase 2 Signal Alignment Summary")
    st.caption("All data from public APIs. Insider data = public SEC filings only. For research only.")

    p2_rows = []
    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty: continue
        data = add_indicators(raw.copy())
        if data.empty: continue
        cs = load_news_cache(ticker)   or st.session_state.get(f"sentiment_{ticker}")
        ci = load_insider_cache(ticker) or st.session_state.get(f"insider_{ticker}")
        sl = cs["sentiment_label"]    if cs else "Not fetched"
        is_ = ci["insider_signal"]   if ci else "Not fetched"
        ts, _ = score_stock(data)
        comb, _ = score_stock(data,
                              news_sentiment=cs,
                              insider_signal=is_ if is_ not in ["Not fetched","—"] else None)
        p2_rows.append({"Ticker": ticker, "Tech Score": f"{ts}/100",
                        "News Sentiment": sl, "Insider Disclosure": is_,
                        "Combined Score": f"{comb}/100",
                        "Signal Alignment": get_signal_short(comb)})

    if p2_rows:
        p2df = pd.DataFrame(p2_rows)
        p2df["_s"] = p2df["Combined Score"].str.replace("/100","").astype(int)
        p2df = p2df.sort_values("_s", ascending=False).drop(columns=["_s"])
        st.dataframe(p2df, use_container_width=True, hide_index=True)
    st.caption(DISCLAIMER)
    st.divider()

    st.subheader(f"Deep Dive: {selected}")
    st.caption(f"Publicly available research data for {selected}. Price may be delayed ~15 min.")
    raw = get_data(selected, period)

    if raw.empty:
        st.error(f"No data for {selected}")
    else:
        data   = add_indicators(raw.copy())
        latest = data.iloc[-1]
        cs     = load_news_cache(selected)    or st.session_state.get(f"sentiment_{selected}")
        ci     = load_insider_cache(selected) or st.session_state.get(f"insider_{selected}")
        isig   = ci["insider_signal"] if ci else None
        score, reasons = score_stock(data, news_sentiment=cs, insider_signal=isig)
        signal = get_signal(score)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Research Score",   f"{score}/100")
        c2.metric("Signal Alignment", get_signal_short(score))
        c3.metric("RSI",              f"{safe_float(latest['RSI']):.1f}")
        c4.metric("Close",            f"${safe_float(latest['Close']):.2f}")

        if score >= 70:   st.success(f"Signal alignment: {signal}")
        elif score >= 50: st.info(f"Signal alignment: {signal}")
        else:             st.warning(f"Signal alignment: {signal}")

        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                      low=data["Low"], close=data["Close"], name="Price"))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(color="orange")))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(color="royalblue")))
        fig2.update_layout(height=500, xaxis_rangeslider_visible=False,
                           title=f"{selected} — Deep Dive (data may be delayed ~15 min)")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Score breakdown (research signals only)"):
            st.caption("Educational signal indicators only. Do not predict future performance.")
            for r in reasons: st.write(f"• {r}")
            st.caption(DISCLAIMER)

        # ── Phase 3 Prompt 2: Auto-save snapshot ─────────────────────────────
        snap_result = save_signal_snapshot(
            ticker               = selected,
            close                = safe_float(latest["Close"]),
            rsi                  = safe_float(latest["RSI"]),
            ma20                 = safe_float(latest["MA20"]),
            ma50                 = safe_float(latest["MA50"]),
            volume               = safe_float(latest["Volume"]),
            technical_score      = score_stock(data)[0],
            news_sentiment_label = cs["sentiment_label"] if cs else None,
            insider_signal       = isig,
            combined_score       = score,
            final_signal         = get_signal_short(score)
        )
        if snap_result == "saved":
            st.success(f"📸 Today's signal snapshot saved for {selected}.")
        else:
            st.caption(f"📸 Snapshot for {selected} already exists for today.")

        # ── News sentiment ────────────────────────────────────────────────────
        with st.expander("News Sentiment — Alpha Vantage (public news data)", expanded=False):
            st.caption("Derived from publicly available news articles. Does not predict price movement.")
            if not alpha_key:
                st.warning("Add ALPHA_VANTAGE_API_KEY to Streamlit Secrets.")
            else:
                if render_request_gate("alpha_vantage", AV_DAILY_LIMIT, selected, "Alpha Vantage", f"av_{selected}"):
                    with st.spinner(f"Fetching public news sentiment for {selected}..."):
                        result, err = fetch_news_sentiment(selected)
                    if err: st.error(f"Could not load sentiment: {err}")
                    else:
                        save_news_cache(selected, result)
                        st.session_state[f"sentiment_{selected}"] = result
                        st.rerun()

                cached = load_news_cache(selected) or st.session_state.get(f"sentiment_{selected}")
                if cached:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Sentiment Lean",  cached["sentiment_label"])
                    s2.metric("Avg Score",       cached["avg_score"])
                    s3.metric("Articles Found",  cached["article_count"])
                    if cached["sentiment_label"] == "Bullish":
                        st.success("News sentiment leaning bullish — signals may be favorable")
                    elif cached["sentiment_label"] == "Bearish":
                        st.error("News sentiment leaning bearish — signals may be unfavorable")
                    else:
                        st.info("News sentiment neutral — no strong directional signal")
                    st.markdown("**Top recent public headlines:**")
                    for a in cached["top_articles"][:3]:
                        t = a["time"]
                        if len(t) == 8: t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
                        st.markdown(f"- [{a['title']}]({a['url']})  \n  *{a['source']} · {t} · {a['sentiment']}*")

                    st.divider()
                    if client:
                        st.markdown("**Analyze articles with AI (uses OpenAI API)**")
                        st.caption("AI summarizes bullish/bearish factors from headlines. Research only.")
                        if render_openai_gate(f"analyze_articles_{selected}", "Analyze News Articles"):
                            increment_usage("openai", 1)
                            with st.spinner("Analyzing articles..."):
                                st.write(analyze_articles_with_ai(selected, cached["top_articles"]))
                            st.caption(DISCLAIMER)

                    st.divider()
                    st.markdown("**Use ChatGPT instead — free, same results**")
                    st.caption("Copy → paste into chatgpt.com. No API cost.")
                    st.code(build_chatgpt_prompt(selected, latest, score, reasons,
                                                  sentiment_data=cached,
                                                  insider_data=load_insider_cache(selected) or
                                                  st.session_state.get(f"insider_{selected}")),
                            language="")
                else:
                    st.caption("No sentiment data loaded yet. Confirm above to fetch.")

        # ── Insider activity ──────────────────────────────────────────────────
        with st.expander("Insider Disclosure Activity — Finnhub (public SEC filings only)", expanded=False):
            st.caption("All data from public regulatory disclosures (SEC Form 4). Not non-public information. "
                       "Insider selling is often routine. Open-market buying is generally a stronger signal.")
            if not finnhub_key:
                st.warning("Add FINNHUB_API_KEY to Streamlit Secrets.")
            else:
                if render_request_gate("finnhub", FINNHUB_DAILY_LIMIT, selected, "Finnhub", f"fh_{selected}"):
                    with st.spinner(f"Fetching public insider disclosures for {selected}..."):
                        ir, ie = fetch_insider_transactions(selected)
                    if ie: st.error(f"Could not load insider data: {ie}")
                    else:
                        save_insider_cache(selected, ir)
                        st.session_state[f"insider_{selected}"] = ir
                        st.rerun()

                ci2 = load_insider_cache(selected) or st.session_state.get(f"insider_{selected}")
                if ci2:
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Disclosure Signal", ci2["insider_signal"])
                    i2.metric("Open-Mkt Buys",     ci2.get("open_buy_count",  ci2["buy_count"]))
                    i3.metric("Open-Mkt Sells",     ci2.get("open_sell_count", ci2["sell_count"]))
                    net = ci2["net_shares"]
                    if ci2["insider_signal"] == "Bullish":
                        st.success(f"Open-market insider buying — net {net:,} shares. Potentially positive signal.")
                    elif ci2["insider_signal"] == "Bearish":
                        st.warning(f"Net insider selling — {net:,} shares. Often routine or planned.")
                    else:
                        st.info(f"Mixed or neutral insider activity. Net shares: {net:,}")
                    st.caption("Open-Market Buy = strongest. Grant/Award = compensation. Planned Sell = often pre-scheduled.")
                    if ci2["transactions"]:
                        st.dataframe(pd.DataFrame(ci2["transactions"]), use_container_width=True, hide_index=True)
                    st.caption(DISCLAIMER)
                else:
                    st.caption("No insider data loaded yet. Confirm above to fetch.")

        # ── AI Summary ───────────────────────────────────────────────────────
        st.subheader("AI Research Summary")
        st.caption("AI summaries are for paper trading research only. AI uses hedged language and never recommends trades.")
        st.markdown("**Option 1 — Free: Copy prompt → paste into chatgpt.com**")
        st.caption("No API cost. Same research quality.")
        st.code(build_chatgpt_prompt(selected, latest, score, reasons,
                                      sentiment_data=cs, insider_data=ci), language="")
        st.markdown("**Option 2 — Use OpenAI API (costs tokens)**")
        if client is None:
            st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
        elif render_openai_gate("ai_deepdive", f"Generate AI summary for {selected}"):
            increment_usage("openai", 1)
            with st.spinner("Generating research summary..."):
                st.info(generate_ai_analysis(selected, latest, score, sentiment_data=cs, insider_data=ci))
            st.caption(DISCLAIMER)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Backtesting (Prompts 3 + 4)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Signal Backtesting")
    st.info(
        "⚠️ **Backtesting disclaimer:** Past signal performance does not guarantee future results. "
        "Returns shown are hypothetical and based on signal snapshots vs current price. "
        "This is for educational research only. Not financial advice."
    )

    history_df = load_signal_history()

    if history_df.empty:
        st.info("No signal history yet. Visit the Deep Dive tab to start saving daily snapshots.")
    else:
        # Calculate returns
        rows = []
        for _, row in history_df.iterrows():
            current = get_current_price(row["ticker"])
            if current and row["close"] > 0:
                ret_pct   = ((current - row["close"]) / row["close"]) * 100
                profitable = ret_pct > 0
            else:
                ret_pct   = None
                profitable = None
            try:
                snap_date = datetime.strptime(row["snapshot_date"], "%Y-%m-%d").date()
                days_held = (date.today() - snap_date).days
            except:
                days_held = 0
            rows.append({
                "Ticker":         row["ticker"],
                "Snapshot Date":  row["snapshot_date"],
                "Signal":         row["final_signal"],
                "Score@Snap":     row["combined_score"],
                "Close@Snap":     row["close"],
                "Current Price":  current,
                "Return %":       round(ret_pct, 2) if ret_pct is not None else None,
                "Days Held":      days_held,
                "Profitable":     profitable,
                "News@Snap":      row["news_sentiment_label"],
                "Insider@Snap":   row["insider_signal"],
            })

        bt_df = pd.DataFrame(rows)
        valid = bt_df.dropna(subset=["Return %"])

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("Overall Backtest Summary")
        st.caption("Hypothetical returns based on snapshot close vs current price. Educational only.")

        if not valid.empty:
            win_rate   = valid["Profitable"].mean() * 100
            avg_return = valid["Return %"].mean()
            best       = valid.loc[valid["Return %"].idxmax()]
            worst      = valid.loc[valid["Return %"].idxmin()]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate",       f"{win_rate:.1f}%")
            m2.metric("Avg Return",     f"{avg_return:+.2f}%")
            m3.metric("Best Signal",    f"{best['Ticker']} {best['Return %']:+.2f}%")
            m4.metric("Worst Signal",   f"{worst['Ticker']} {worst['Return %']:+.2f}%")

            st.markdown("**Average return by signal type:**")
            st.caption("Do bullish signals historically outperform neutral or weak ones in your research?")
            sig_avg = (valid.groupby("Signal")["Return %"]
                       .agg(["mean", "count"])
                       .reset_index()
                       .rename(columns={"mean": "Avg Return %", "count": "# Signals"}))
            sig_avg["Avg Return %"] = sig_avg["Avg Return %"].round(2)
            st.dataframe(sig_avg, use_container_width=True, hide_index=True)

        st.subheader("Full Signal History")
        display_df = bt_df.copy()
        display_df["Return %"] = display_df["Return %"].apply(
            lambda x: f"{x:+.2f}%" if x is not None else "Pending"
        )
        display_df["Current Price"] = display_df["Current Price"].apply(
            lambda x: f"${x:.2f}" if x else "N/A"
        )
        display_df["Close@Snap"] = display_df["Close@Snap"].apply(lambda x: f"${x:.2f}")
        st.dataframe(display_df.drop(columns=["Profitable"]),
                     use_container_width=True, hide_index=True)
        st.caption(DISCLAIMER)

        # ── Prompt 4: Backtesting visuals ─────────────────────────────────────
        if not valid.empty:
            st.divider()
            st.subheader("Backtest Charts")
            st.caption("All charts are educational. Based on limited historical snapshots — interpret carefully.")

            # Chart 1: Avg return by score range
            st.markdown("**Average return by research score range**")
            st.caption("Do higher research scores correlate with better hypothetical returns in your data?")
            valid2 = valid.copy()
            valid2["Score Range"] = pd.cut(
                valid2["Score@Snap"],
                bins=[0, 40, 55, 70, 100],
                labels=["0–40 (Weak)", "41–55 (Neutral-Low)", "56–70 (Neutral-High)", "71–100 (Bullish)"]
            )
            sr_avg = valid2.groupby("Score Range", observed=True)["Return %"].mean().reset_index()
            fig_sr = px.bar(sr_avg, x="Score Range", y="Return %",
                            title="Avg Hypothetical Return by Score Range",
                            color="Return %", color_continuous_scale="RdYlGn")
            fig_sr.update_layout(height=350)
            st.plotly_chart(fig_sr, use_container_width=True)

            # Chart 2: Win rate by signal type
            st.markdown("**Win rate by signal type**")
            st.caption("What percentage of each signal type resulted in a positive hypothetical return?")
            wr_sig = (valid.groupby("Signal")["Profitable"]
                      .agg(lambda x: x.mean() * 100)
                      .reset_index()
                      .rename(columns={"Profitable": "Win Rate %"}))
            fig_wr = px.bar(wr_sig, x="Signal", y="Win Rate %",
                            title="Win Rate % by Signal Type",
                            color="Signal",
                            color_discrete_map={"Bullish": "green", "Neutral": "gold", "Weak": "red"})
            fig_wr.update_layout(height=350)
            st.plotly_chart(fig_wr, use_container_width=True)

            # Chart 3: Signal distribution
            st.markdown("**Distribution of signals saved**")
            st.caption("How many of your saved snapshots were Bullish, Neutral, or Weak?")
            sig_dist = bt_df["Signal"].value_counts().reset_index()
            sig_dist.columns = ["Signal", "Count"]
            fig_dist = px.pie(sig_dist, names="Signal", values="Count",
                              title="Signal Distribution",
                              color="Signal",
                              color_discrete_map={"Bullish": "green", "Neutral": "gold", "Weak": "red"})
            st.plotly_chart(fig_dist, use_container_width=True)

            # Chart 4 + 5: Top and worst performers
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Top performing tickers**")
                st.caption("Tickers with the highest hypothetical returns since snapshot.")
                top5 = valid.nlargest(5, "Return %")[["Ticker", "Signal", "Return %"]]
                fig_top = px.bar(top5, x="Ticker", y="Return %",
                                 color="Return %", color_continuous_scale="Greens",
                                 title="Top 5 Hypothetical Returns")
                fig_top.update_layout(height=300)
                st.plotly_chart(fig_top, use_container_width=True)

            with col_b:
                st.markdown("**Worst performing tickers**")
                st.caption("Tickers with the lowest hypothetical returns since snapshot.")
                bot5 = valid.nsmallest(5, "Return %")[["Ticker", "Signal", "Return %"]]
                fig_bot = px.bar(bot5, x="Ticker", y="Return %",
                                 color="Return %", color_continuous_scale="Reds_r",
                                 title="Bottom 5 Hypothetical Returns")
                fig_bot.update_layout(height=300)
                st.plotly_chart(fig_bot, use_container_width=True)

            st.caption(DISCLAIMER)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stock Discovery (Prompts 5 + 6)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stock Discovery Scanner")
    st.info(
        "📡 This scanner uses **yfinance only** — no Alpha Vantage, Finnhub, or OpenAI calls. "
        "Scan freely without worrying about API limits. "
        "After finding a candidate, run Phase 2 analysis manually in the Deep Dive tab. "
        "For educational research only. Not financial advice."
    )

    category = st.selectbox("Choose a category to scan", list(DISCOVERY_LISTS.keys()))
    scan_tickers = DISCOVERY_LISTS[category]

    st.caption(f"Scanning {len(scan_tickers)} tickers in **{category}**: {', '.join(scan_tickers)}")
    st.caption("Uses yfinance only. No paid API calls.")

    if st.button(f"Scan {category}", key="run_scan"):
        scan_rows = []
        prog = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(scan_tickers):
            status_text.caption(f"Scanning {ticker} ({i+1}/{len(scan_tickers)})...")
            prog.progress((i + 1) / len(scan_tickers))
            try:
                raw = get_data(ticker, "3mo")
                if raw.empty: continue
                data = add_indicators(raw.copy())
                if data.empty: continue
                latest = data.iloc[-1]
                score, reasons = score_stock(data)
                signal = get_signal_short(score)
                top_reason = reasons[0] if reasons else "—"
                scan_rows.append({
                    "Ticker":          ticker,
                    "Close":           f"${safe_float(latest['Close']):.2f}",
                    "RSI":             f"{safe_float(latest['RSI']):.1f}",
                    "MA20":            f"${safe_float(latest['MA20']):.2f}",
                    "MA50":            f"${safe_float(latest['MA50']):.2f}",
                    "Volume":          f"{safe_float(latest['Volume']):,.0f}",
                    "Research Score":  score,
                    "Signal":          signal,
                    "Top Signal":      top_reason,
                })
            except Exception as e:
                st.caption(f"Could not scan {ticker}: {e}")

        prog.empty()
        status_text.empty()

        if scan_rows:
            scan_df = pd.DataFrame(scan_rows).sort_values("Research Score", ascending=False)
            scan_df["Research Score"] = scan_df["Research Score"].apply(lambda x: f"{x}/100")
            st.session_state["scan_results"] = scan_rows
            st.session_state["scan_category"] = category

    # Display results
    if "scan_results" in st.session_state:
        scan_rows = st.session_state["scan_results"]
        scan_df   = (pd.DataFrame(scan_rows)
                     .sort_values("Research Score", ascending=False)
                     .reset_index(drop=True))

        st.subheader(f"Scan Results — {st.session_state.get('scan_category', category)}")
        st.caption(
            "Ranked by research score (highest = most technically aligned). "
            "Score is an educational metric only. Not a recommendation."
        )

        display_df = scan_df.copy()
        display_df["Research Score"] = display_df["Research Score"].apply(lambda x: f"{x}/100")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(DISCLAIMER)

        # ── Prompt 6: Send to Deep Dive ───────────────────────────────────────
        st.divider()
        st.subheader("Send a Candidate to Deep Dive")
        st.caption(
            "Select a ticker from the scan results to analyze further. "
            "After selecting, go to the **Deep Dive + Phase 2** tab to run news sentiment, "
            "insider activity, and AI summary manually."
        )

        candidate_tickers = scan_df["Ticker"].tolist()
        chosen = st.selectbox("Select a ticker to investigate further", candidate_tickers)

        col_a, col_b = st.columns(2)

        if col_a.button(f"Set {chosen} as Deep Dive ticker"):
            current_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if chosen not in current_tickers:
                new_input = tickers_input.rstrip(", ") + f", {chosen}"
                st.success(
                    f"**{chosen}** added to your watchlist. "
                    f"Update the ticker input in the sidebar to: `{new_input}` "
                    f"then select **{chosen}** in the Deep Dive dropdown."
                )
            else:
                st.info(f"{chosen} is already in your watchlist. Select it in the Deep Dive tab.")

        if col_b.button(f"Show quick stats for {chosen}"):
            raw = get_data(chosen, "3mo")
            if not raw.empty:
                data   = add_indicators(raw.copy())
                latest = data.iloc[-1]
                score, reasons = score_stock(data)
                st.markdown(f"**{chosen} Quick Stats**")
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Close",          f"${safe_float(latest['Close']):.2f}")
                q2.metric("RSI",            f"{safe_float(latest['RSI']):.1f}")
                q3.metric("Research Score", f"{score}/100")
                q4.metric("Signal",         get_signal_short(score))
                st.markdown("Signal breakdown:")
                for r in reasons: st.write(f"• {r}")
                st.caption(DISCLAIMER)

        st.info(
            f"**Next steps for {chosen}:**\n"
            "1. Add it to your watchlist in the sidebar\n"
            "2. Go to **Deep Dive + Phase 2** tab\n"
            "3. Fetch news sentiment (costs 1 Alpha Vantage request)\n"
            "4. Fetch insider disclosures (costs 1 Finnhub request)\n"
            "5. Use the free ChatGPT prompt or OpenAI button for AI analysis\n"
            "6. If setup looks interesting, log a paper trade in the Paper Trade Log tab\n\n"
            "⚠️ All analysis is for educational research only. Not financial advice."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Paper Trade Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Paper Trade Log")
    st.info(
        "📋 **Paper Trading Only:** All trades logged here are hypothetical and for educational "
        "purposes only. No real money is involved. Not financial advice."
    )

    with st.expander("Log a new paper trade"):
        st.caption("Track hypothetical trades to see how your research signals perform over time.")
        c1, c2, c3 = st.columns(3)
        trade_ticker = c1.text_input("Ticker", value=selected)
        trade_action = c2.selectbox("Action", ["BUY (paper)", "SELL (paper)"])
        trade_action_clean = trade_action.replace(" (paper)", "")

        raw_log = get_data(trade_ticker, "5d")
        default_price = float(raw_log["Close"].iloc[-1]) if not raw_log.empty else 100.0
        trade_price   = c3.number_input("Hypothetical Entry Price", value=default_price)

        raw_ind = get_data(trade_ticker, period)
        if not raw_ind.empty:
            ind_data       = add_indicators(raw_ind.copy())
            ind_latest     = ind_data.iloc[-1]
            trade_score, _ = score_stock(ind_data)
            trade_signal   = get_signal_short(trade_score)
            trade_rsi      = safe_float(ind_latest["RSI"])
            trade_ma20     = safe_float(ind_latest["MA20"])
            trade_ma50     = safe_float(ind_latest["MA50"])
        else:
            trade_score = 0; trade_signal = "Unknown"
            trade_rsi = 0.0; trade_ma20 = 0.0; trade_ma50 = 0.0

        trade_notes = st.text_input("Research notes (optional)")
        st.caption(f"Will save with: Score {trade_score}/100 | RSI {trade_rsi:.1f} | Signal: {trade_signal}")

        if st.button("Log Paper Trade"):
            save_trade(
                entry_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                ticker=trade_ticker, action=trade_action_clean,
                entry_price=trade_price, ai_score=trade_score,
                rsi=trade_rsi, ma20=trade_ma20, ma50=trade_ma50,
                signal=trade_signal, notes=trade_notes
            )
            st.success(f"Paper trade logged: {trade_action_clean} {trade_ticker} at ${trade_price:.2f} (hypothetical)")
            st.rerun()

    trades_df = load_trades()

    if trades_df.empty:
        st.info("No paper trades logged yet.")
    else:
        st.subheader("Open Paper Trades")
        st.caption("Gain/Loss is hypothetical. For educational tracking only.")
        open_trades = trades_df[trades_df["status"] == "Open"].copy()

        if not open_trades.empty:
            perf_rows = []
            for _, row in open_trades.iterrows():
                current_price = get_current_price(row["ticker"])
                if current_price and row["entry_price"] > 0:
                    gain_pct = ((current_price - row["entry_price"]) / row["entry_price"]) * 100
                    if row["action"] == "SELL": gain_pct = -gain_pct
                    gain_str = f"{gain_pct:+.2f}% (hypothetical)"
                else:
                    current_price = 0.0; gain_str = "N/A"
                try:
                    days_open = (datetime.now() - datetime.strptime(row["entry_date"], "%Y-%m-%d %H:%M")).days
                except:
                    days_open = 0
                perf_rows.append({
                    "ID": row["id"], "Date": row["entry_date"],
                    "Ticker": row["ticker"], "Action": row["action"],
                    "Entry $": f"${row['entry_price']:.2f}",
                    "Current $": f"${current_price:.2f}",
                    "Hypothetical P/L": gain_str, "Days Open": days_open,
                    "Score@Entry": f"{row['ai_score']}/100",
                    "RSI@Entry": f"{row['rsi']:.1f}",
                    "Signal@Entry": row["signal"], "Notes": row["notes"]
                })

            perf_df = pd.DataFrame(perf_rows)
            st.dataframe(perf_df.drop(columns=["ID"]), use_container_width=True, hide_index=True)
            st.caption(DISCLAIMER)

            st.subheader("Manage paper trades")
            trade_ids    = open_trades["id"].tolist()
            trade_labels = [
                f"{r['ticker']} {r['action']} @ ${r['entry_price']:.2f} ({r['entry_date']})"
                for _, r in open_trades.iterrows()
            ]
            sel_label  = st.selectbox("Select a trade", trade_labels)
            sel_id     = trade_ids[trade_labels.index(sel_label)]
            mc1, mc2   = st.columns(2)
            if mc1.button("Mark as Closed"):
                close_trade(sel_id); st.success("Marked as closed."); st.rerun()
            if mc2.button("Delete Trade"):
                delete_trade(sel_id); st.success("Deleted."); st.rerun()

        closed_trades = trades_df[trades_df["status"] == "Closed"]
        if not closed_trades.empty:
            st.subheader("Closed Paper Trades")
            st.dataframe(
                closed_trades[["entry_date","ticker","action","entry_price",
                               "ai_score","rsi","signal","notes"]].rename(columns={
                    "entry_date": "Date", "ticker": "Ticker", "action": "Action",
                    "entry_price": "Entry $", "ai_score": "Score@Entry",
                    "rsi": "RSI@Entry", "signal": "Signal@Entry", "notes": "Notes"
                }), use_container_width=True, hide_index=True)
            st.caption(DISCLAIMER)

    st.divider()
    st.caption(
        "📋 All entries are hypothetical paper trades for educational research only. "
        "No real money is tracked here. Not financial advice."
    )
