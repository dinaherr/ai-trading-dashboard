import yfinance as yf
import pandas as pd
import streamlit as st

# ── Price data ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_data(ticker, period):
    df = yf.download(
        ticker, period=period, interval="1d",
        progress=False, auto_adjust=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_current_price(ticker):
    try:
        df = yf.download(
            ticker, period="2d", interval="1d",
            progress=False, auto_adjust=True
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except:
        return None
