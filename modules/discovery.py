# ── Preset watchlists ─────────────────────────────────────────────────────────
DISCOVERY_LISTS = {
    "AI & Machine Learning": [
        "NVDA", "AMD", "MSFT", "GOOGL", "META",
        "AMZN", "ORCL", "IBM", "PLTR", "AI",
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "FTNT", "ZS", "S",
        "OKTA", "CYBR", "TENB", "RPD", "SAIL",
    ],
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO",
        "MU", "AMAT", "LRCX", "KLAC", "TSM",
    ],
    "Defense": [
        "LMT", "RTX", "NOC", "GD", "BA",
        "HII", "LDOS", "CACI", "SAIC", "KTOS",
    ],
    "Biotech": [
        "MRNA", "BNTX", "REGN", "VRTX", "BIIB",
        "GILD", "AMGN", "ILMN", "RARE", "EXAS",
    ],
    "Cloud & SaaS": [
        "CRM", "NOW", "SNOW", "DDOG", "MDB",
        "NET", "HUBS", "ZM", "TEAM", "WDAY",
    ],
    "Major ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "XLK",
        "XLF", "XLV", "GLD", "TLT", "VNQ",
    ],
    "Mega-Cap Tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "NFLX", "ADBE", "CRM",
    ],
}


# ── Category helpers ──────────────────────────────────────────────────────────
def get_category_names():
    return list(DISCOVERY_LISTS.keys())

def get_category_tickers(category):
    return DISCOVERY_LISTS.get(category, [])


# ── Scanner ───────────────────────────────────────────────────────────────────
def scan_category(category, get_data_fn, add_indicators_fn,
                  score_stock_fn, get_signal_short_fn, safe_float_fn,
                  period="3mo"):
    """
    Scans all tickers in a preset category using yfinance only.
    No paid API calls triggered here.
    Returns list sorted by Research Score descending.
    """
    tickers = DISCOVERY_LISTS.get(category, [])
    results = []

    for ticker in tickers:
        try:
            raw = get_data_fn(ticker, period)
            if raw.empty:
                continue
            data = add_indicators_fn(raw.copy())
            if data.empty:
                continue
            latest         = data.iloc[-1]
            score, reasons = score_stock_fn(data)
            signal         = get_signal_short_fn(score)

            results.append({
                "Ticker":         ticker,
                "Close":          f"${safe_float_fn(latest['Close']):.2f}",
                "RSI":            f"{safe_float_fn(latest['RSI']):.1f}",
                "MA20":           f"${safe_float_fn(latest['MA20']):.2f}",
                "MA50":           f"${safe_float_fn(latest['MA50']):.2f}",
                "Volume":         f"{safe_float_fn(latest['Volume']):,.0f}",
                "Research Score": score,
                "Signal":         signal,
                "Top Signal":     reasons[0] if reasons else "—",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["Research Score"], reverse=True)
    return results


# ── Display dataframe ─────────────────────────────────────────────────────────
def build_display_df(scan_results):
    """
    Converts raw scan results into a display-ready DataFrame.
    Formats Research Score as "N/100" string.
    """
    import pandas as pd
    df = (
        pd.DataFrame(scan_results)
        .sort_values("Research Score", ascending=False)
        .reset_index(drop=True)
    )
    df["Research Score"] = df["Research Score"].apply(lambda x: f"{x}/100")
    return df


# ── Quick stats ───────────────────────────────────────────────────────────────
def get_quick_stats(ticker, get_data_fn, add_indicators_fn,
                    score_stock_fn, get_signal_short_fn,
                    safe_float_fn, period="3mo"):
    """
    Fetches a single ticker's technical stats for quick preview.
    Uses yfinance only — no paid APIs triggered.

    Returns dict with: ticker, close, rsi, score, signal, reasons, latest, data
    Returns None if data unavailable.
    """
    try:
        raw = get_data_fn(ticker, period)
        if raw.empty:
            return None
        data = add_indicators_fn(raw.copy())
        if data.empty:
            return None
        latest         = data.iloc[-1]
        score, reasons = score_stock_fn(data)
        signal         = get_signal_short_fn(score)

        return {
            "ticker":  ticker,
            "close":   safe_float_fn(latest["Close"]),
            "rsi":     safe_float_fn(latest["RSI"]),
            "score":   score,
            "signal":  signal,
            "reasons": reasons,
            "latest":  latest,
            "data":    data,
        }
    except Exception:
        return None
