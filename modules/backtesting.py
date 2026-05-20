import pandas as pd
from datetime import date, datetime

# ── Build backtest rows ───────────────────────────────────────────────────────
def build_backtest_rows(history_df, get_current_price_fn):
    """
    Compares each saved signal snapshot's close price to current price.
    Calculates hypothetical return, days held, and profitability.

    Args:
        history_df:           pandas DataFrame from load_signal_history
        get_current_price_fn: callable — get_current_price from market_data module

    Returns:
        pandas DataFrame with all backtest columns
    """
    rows = []
    for _, row in history_df.iterrows():
        current = get_current_price_fn(row["ticker"])
        if current and row["close"] > 0:
            ret_pct    = ((current - row["close"]) / row["close"]) * 100
            profitable = ret_pct > 0
        else:
            ret_pct    = None
            profitable = None

        try:
            days_held = (
                date.today() -
                datetime.strptime(row["snapshot_date"], "%Y-%m-%d").date()
            ).days
        except:
            days_held = 0

        rows.append({
            "Ticker":        row["ticker"],
            "Snapshot Date": row["snapshot_date"],
            "Signal":        row["final_signal"],
            "Score@Snap":    row["combined_score"],
            "Close@Snap":    row["close"],
            "Current Price": current,
            "Return %":      round(ret_pct, 2) if ret_pct is not None else None,
            "Days Held":     days_held,
            "Profitable":    profitable,
            "News@Snap":     row["news_sentiment_label"],
            "Insider@Snap":  row["insider_signal"],
        })

    return pd.DataFrame(rows)


# ── Summary metrics ───────────────────────────────────────────────────────────
def compute_summary(bt_df):
    """
    Computes win rate, average return, best and worst signal from backtest data.

    Args:
        bt_df: pandas DataFrame from build_backtest_rows

    Returns:
        dict with win_rate, avg_return, best, worst
        Returns None if no valid rows
    """
    valid = bt_df.dropna(subset=["Return %"])
    if valid.empty:
        return None

    return {
        "win_rate":   valid["Profitable"].mean() * 100,
        "avg_return": valid["Return %"].mean(),
        "best":       valid.loc[valid["Return %"].idxmax()],
        "worst":      valid.loc[valid["Return %"].idxmin()],
        "valid":      valid,
    }


# ── Average return by signal ──────────────────────────────────────────────────
def signal_avg_returns(valid_df):
    """
    Groups valid backtest rows by signal type and calculates average return.

    Args:
        valid_df: pandas DataFrame — rows with non-null Return %

    Returns:
        pandas DataFrame with Signal, Avg Return %, # Signals columns
    """
    df = (
        valid_df.groupby("Signal")["Return %"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg Return %", "count": "# Signals"})
    )
    df["Avg Return %"] = df["Avg Return %"].round(2)
    return df


# ── Score range bucketing ─────────────────────────────────────────────────────
def bucket_by_score_range(valid_df):
    """
    Bins scores into ranges and calculates average return per range.

    Args:
        valid_df: pandas DataFrame — rows with non-null Return %

    Returns:
        pandas DataFrame with Score Range and Return % columns
    """
    df = valid_df.copy()
    df["Score Range"] = pd.cut(
        df["Score@Snap"],
        bins=[0, 40, 55, 70, 100],
        labels=[
            "0–40 (Weak)",
            "41–55 (Neutral-Low)",
            "56–70 (Neutral-High)",
            "71–100 (Bullish)",
        ]
    )
    return (
        df.groupby("Score Range", observed=True)["Return %"]
        .mean()
        .reset_index()
    )


# ── Display formatting ────────────────────────────────────────────────────────
def format_display_df(bt_df):
    """
    Formats the raw backtest DataFrame for display in Streamlit.
    Drops the internal Profitable column.

    Args:
        bt_df: pandas DataFrame from build_backtest_rows

    Returns:
        pandas DataFrame ready for st.dataframe
    """
    df = bt_df.copy()
    df["Return %"]      = df["Return %"].apply(
        lambda x: f"{x:+.2f}%" if x is not None else "Pending"
    )
    df["Current Price"] = df["Current Price"].apply(
        lambda x: f"${x:.2f}" if x else "N/A"
    )
    df["Close@Snap"]    = df["Close@Snap"].apply(lambda x: f"${x:.2f}")
    return df.drop(columns=["Profitable"])
