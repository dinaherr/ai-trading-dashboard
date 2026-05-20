# Database functions will go here
import sqlite3
import pandas as pd
from datetime import date
import json

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH = "data/trades.db"

AV_DAILY_LIMIT      = 25
FINNHUB_DAILY_LIMIT = 60
OPENAI_DAILY_LIMIT  = 20
SEC_DAILY_LIMIT     = 50

# ── Init ──────────────────────────────────────────────────────────────────────
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
    c.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, snapshot_date TEXT, close REAL,
            rsi REAL, ma20 REAL, ma50 REAL, volume REAL,
            technical_score INTEGER, news_sentiment_label TEXT,
            insider_signal TEXT, combined_score INTEGER,
            final_signal TEXT, notes TEXT,
            UNIQUE(ticker, snapshot_date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sec_company_cache (
            ticker TEXT PRIMARY KEY,
            cik TEXT, company_name TEXT,
            submissions_url TEXT, fetched_date TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings_cache (
            ticker TEXT, fetched_date TEXT, filings_json TEXT,
            PRIMARY KEY (ticker, fetched_date)
        )
    """)
    conn.commit()
    conn.close()

# ── Trades ────────────────────────────────────────────────────────────────────
def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()
    return df

def save_trade(entry_date, ticker, action, entry_price,
               ai_score, rsi, ma20, ma50, signal, notes):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO trades
        (entry_date,ticker,action,entry_price,ai_score,rsi,ma20,ma50,signal,notes,status)
        VALUES (?,?,?,?,?,?,?,?,?,?,'Open')
    """, (entry_date, ticker, action, entry_price,
          ai_score, rsi, ma20, ma50, signal, notes))
    conn.commit()
    conn.close()

def close_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute(
        "UPDATE trades SET status='Closed' WHERE id=?", (trade_id,)
    )
    conn.commit()
    conn.close()

def delete_trade(trade_id):
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("DELETE FROM trades WHERE id=?", (trade_id,))
    conn.commit()
    conn.close()

# ── API usage ─────────────────────────────────────────────────────────────────
def get_usage_today(api_name):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute(
        "SELECT count FROM api_usage WHERE api_name=? AND usage_date=?",
        (api_name, today)
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_usage(api_name, amount=1):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute(
        "SELECT count FROM api_usage WHERE api_name=? AND usage_date=?",
        (api_name, today)
    )
    row = c.fetchone()
    if row:
        c.execute(
            "UPDATE api_usage SET count=count+? WHERE api_name=? AND usage_date=?",
            (amount, api_name, today)
        )
    else:
        c.execute(
            "INSERT INTO api_usage (api_name,usage_date,count) VALUES (?,?,?)",
            (api_name, today, amount)
        )
    conn.commit()
    conn.close()

def requests_remaining(api_name, limit):
    return max(0, limit - get_usage_today(api_name))

# ── News cache ────────────────────────────────────────────────────────────────
def save_news_cache(ticker, result):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        INSERT OR REPLACE INTO news_cache
        (ticker,fetched_date,article_count,avg_score,sentiment_label,top_articles)
        VALUES (?,?,?,?,?,?)
    """, (ticker, today, result["article_count"], result["avg_score"],
          result["sentiment_label"], json.dumps(result["top_articles"])))
    conn.commit()
    conn.close()

def load_news_cache(ticker):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("""
        SELECT article_count,avg_score,sentiment_label,top_articles
        FROM news_cache WHERE ticker=? AND fetched_date=?
    """, (ticker, today))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "article_count":   row[0],
            "avg_score":       row[1],
            "sentiment_label": row[2],
            "top_articles":    json.loads(row[3]),
        }
    return None

# ── Insider cache ─────────────────────────────────────────────────────────────
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
    conn.commit()
    conn.close()

def load_insider_cache(ticker):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("""
        SELECT buy_count,sell_count,net_shares,insider_signal,transactions
        FROM insider_cache WHERE ticker=? AND fetched_date=?
    """, (ticker, today))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "buy_count":      row[0],
            "sell_count":     row[1],
            "net_shares":     row[2],
            "insider_signal": row[3],
            "transactions":   json.loads(row[4]),
        }
    return None

# ── Signal history ────────────────────────────────────────────────────────────
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
    df   = pd.read_sql(
        "SELECT * FROM signal_history ORDER BY snapshot_date DESC", conn
    )
    conn.close()
    return df

# ── SEC company cache ─────────────────────────────────────────────────────────
def save_sec_company_cache(ticker, cik, company_name, submissions_url):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        INSERT OR REPLACE INTO sec_company_cache
        (ticker, cik, company_name, submissions_url, fetched_date)
        VALUES (?,?,?,?,?)
    """, (ticker, cik, company_name, submissions_url, today))
    conn.commit()
    conn.close()

def load_sec_company_cache(ticker):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT cik, company_name, submissions_url, fetched_date
        FROM sec_company_cache WHERE ticker=?
    """, (ticker,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "cik":             row[0],
            "company_name":    row[1],
            "submissions_url": row[2],
            "fetched_date":    row[3],
        }
    return None

# ── SEC filings cache ─────────────────────────────────────────────────────────
def save_sec_filings_cache(ticker, filings):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        INSERT OR REPLACE INTO sec_filings_cache
        (ticker, fetched_date, filings_json)
        VALUES (?,?,?)
    """, (ticker, today, json.dumps(filings)))
    conn.commit()
    conn.close()

def load_sec_filings_cache(ticker):
    today = date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("""
        SELECT filings_json FROM sec_filings_cache
        WHERE ticker=? AND fetched_date=?
    """, (ticker, today))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None
