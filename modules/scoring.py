import pandas as pd
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_float(val):
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    return float(val)

# ── Indicators ────────────────────────────────────────────────────────────────
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = -delta.where(delta < 0, 0).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast   = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow   = data["Close"].ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(data, window=20, num_std=2):
    sma        = data["Close"].rolling(window).mean()
    std        = data["Close"].rolling(window).std()
    upper      = sma + (std * num_std)
    lower      = sma - (std * num_std)
    pct_b      = (data["Close"] - lower) / (upper - lower)
    return upper, sma, lower, pct_b

def calculate_atr(data, window=14):
    high_low   = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close  = (data["Low"]  - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()

def add_indicators(df):
    # Trend MAs — kept for compatibility
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Swing EMAs — faster signals
    df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # RSI
    df["RSI"] = calculate_rsi(df)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df)

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"], df["BB_PctB"] = calculate_bollinger(df)

    # ATR — volatility measure for stop/target sizing
    df["ATR"] = calculate_atr(df)

    # Volume
    df["VolumeAvg"]      = df["Volume"].rolling(20).mean()
    df["RelativeVolume"] = df["Volume"] / df["VolumeAvg"]

    # 52-week high/low distance
    df["High52w"] = df["High"].rolling(252).max()
    df["Low52w"]  = df["Low"].rolling(252).min()
    df["PctFrom52wHigh"] = (df["Close"] - df["High52w"]) / df["High52w"] * 100
    df["PctFrom52wLow"]  = (df["Close"] - df["Low52w"])  / df["Low52w"]  * 100

    return df.dropna()


# ── Swing trading score ───────────────────────────────────────────────────────
def score_stock(df, news_sentiment=None, insider_signal=None):
    """
    Swing trading focused scoring system.
    Looks for: momentum alignment, pullback entries, volume confirmation,
    trend structure, and volatility context.
    Max possible score: 100
    """
    score   = 50
    reasons = []
    latest  = df.iloc[-1]

    price        = safe_float(latest["Close"])
    ema9         = safe_float(latest["EMA9"])
    ema21        = safe_float(latest["EMA21"])
    ma50         = safe_float(latest["MA50"])
    rsi          = safe_float(latest["RSI"])
    macd         = safe_float(latest["MACD"])
    macd_signal  = safe_float(latest["MACD_Signal"])
    macd_hist    = safe_float(latest["MACD_Hist"])
    bb_pctb      = safe_float(latest["BB_PctB"])
    bb_upper     = safe_float(latest["BB_Upper"])
    bb_lower     = safe_float(latest["BB_Lower"])
    rel_vol      = safe_float(latest["RelativeVolume"])
    atr          = safe_float(latest["ATR"])
    pct_52w_high = safe_float(latest["PctFrom52wHigh"])
    pct_52w_low  = safe_float(latest["PctFrom52wLow"])

    # ── 1. EMA trend alignment (+15) ─────────────────────────────────────────
    # EMA9 above EMA21 = short-term momentum up — primary swing signal
    if ema9 > ema21:
        score += 15
        reasons.append(f"EMA9 (${ema9:.2f}) above EMA21 (${ema21:.2f}) — bullish momentum alignment (+15)")
    else:
        score -= 10
        reasons.append(f"EMA9 below EMA21 — bearish short-term momentum (-10)")

    # ── 2. Price vs MA50 trend filter (+10) ──────────────────────────────────
    # Swing trades work best in the direction of the bigger trend
    if price > ma50:
        score += 10
        reasons.append(f"Price above 50-day MA (${ma50:.2f}) — trading with the trend (+10)")
    else:
        score -= 10
        reasons.append(f"Price below 50-day MA — trading against the trend (-10)")

    # ── 3. RSI swing zone (+15) ───────────────────────────────────────────────
    # Swing entries: RSI 40-60 = healthy pullback or breakout setup
    # RSI 30-40 = oversold bounce potential
    # RSI >75 = overbought, avoid chasing
    if 40 <= rsi <= 60:
        score += 15
        reasons.append(f"RSI {rsi:.1f} — in swing entry zone 40–60, healthy setup (+15)")
    elif 30 <= rsi < 40:
        score += 8
        reasons.append(f"RSI {rsi:.1f} — oversold territory, potential bounce candidate (+8)")
    elif 60 < rsi <= 70:
        score += 5
        reasons.append(f"RSI {rsi:.1f} — mildly elevated, still workable (+5)")
    elif rsi > 75:
        score -= 15
        reasons.append(f"RSI {rsi:.1f} — overbought, high risk of reversal (-15)")
    elif rsi < 30:
        score -= 5
        reasons.append(f"RSI {rsi:.1f} — deeply oversold, wait for stabilization (-5)")
    else:
        reasons.append(f"RSI {rsi:.1f} — neutral zone (no change)")

    # ── 4. MACD momentum (+15) ────────────────────────────────────────────────
    # MACD above signal = bullish momentum
    # Histogram positive and growing = acceleration
    if macd > macd_signal:
        if macd_hist > 0:
            score += 15
            reasons.append(f"MACD above signal line, histogram positive — momentum accelerating (+15)")
        else:
            score += 8
            reasons.append(f"MACD above signal line — bullish momentum (+8)")
    else:
        if macd_hist < 0:
            score -= 10
            reasons.append(f"MACD below signal line, histogram negative — momentum declining (-10)")
        else:
            score -= 5
            reasons.append(f"MACD below signal line — bearish momentum (-5)")

    # ── 5. Bollinger Band position (+10) ──────────────────────────────────────
    # PctB 0.2-0.5 = lower half, potential bounce or pullback entry
    # PctB 0.5-0.8 = upper half, momentum continuation
    # PctB >1.0 = outside upper band, overextended
    # PctB <0.0 = outside lower band, capitulation or breakdown
    if 0.2 <= bb_pctb <= 0.5:
        score += 10
        reasons.append(f"Price in lower Bollinger Band range (PctB {bb_pctb:.2f}) — pullback entry zone (+10)")
    elif 0.5 < bb_pctb <= 0.8:
        score += 5
        reasons.append(f"Price in upper Bollinger Band range (PctB {bb_pctb:.2f}) — momentum zone (+5)")
    elif bb_pctb > 1.0:
        score -= 10
        reasons.append(f"Price above upper Bollinger Band — overextended, risk of snapback (-10)")
    elif bb_pctb < 0.0:
        score -= 8
        reasons.append(f"Price below lower Bollinger Band — breakdown or extreme oversold (-8)")
    else:
        reasons.append(f"Bollinger Band position neutral (PctB {bb_pctb:.2f}) (no change)")

    # ── 6. Volume confirmation (+10) ─────────────────────────────────────────
    # Relative volume >1.5x on a bullish setup confirms conviction
    if rel_vol >= 2.0:
        score += 10
        reasons.append(f"Relative volume {rel_vol:.1f}x average — strong conviction (+10)")
    elif rel_vol >= 1.5:
        score += 7
        reasons.append(f"Relative volume {rel_vol:.1f}x average — above average conviction (+7)")
    elif rel_vol >= 1.0:
        score += 3
        reasons.append(f"Relative volume {rel_vol:.1f}x average — normal volume (+3)")
    else:
        score -= 5
        reasons.append(f"Relative volume {rel_vol:.1f}x average — below average, weak conviction (-5)")

    # ── 7. Distance from 52-week high/low context ─────────────────────────────
    # Near 52w high with momentum = breakout candidate
    # Near 52w low = potential value but needs confirmation
    if -5 <= pct_52w_high <= 0:
        score += 5
        reasons.append(f"{abs(pct_52w_high):.1f}% from 52-week high — near breakout zone (+5)")
    elif -15 <= pct_52w_high < -5:
        score += 3
        reasons.append(f"{abs(pct_52w_high):.1f}% from 52-week high — within swing range (+3)")
    elif pct_52w_high < -40:
        score -= 5
        reasons.append(f"{abs(pct_52w_high):.1f}% below 52-week high — significant drawdown (-5)")

    # ── ATR context (informational only — no score adjustment) ───────────────
    atr_pct = (atr / price) * 100 if price > 0 else 0
    if atr_pct > 5:
        reasons.append(f"ATR {atr:.2f} ({atr_pct:.1f}% of price) — high volatility, size positions carefully")
    elif atr_pct > 2:
        reasons.append(f"ATR {atr:.2f} ({atr_pct:.1f}% of price) — moderate volatility, normal swing range")
    else:
        reasons.append(f"ATR {atr:.2f} ({atr_pct:.1f}% of price) — low volatility, tighter swings")

    # ── Phase 2 signals ───────────────────────────────────────────────────────
    if news_sentiment:
        label = news_sentiment.get("sentiment_label", "Neutral")
        avg   = news_sentiment.get("avg_score", 0)
        if label == "Bullish":
            score += 8
            reasons.append(f"News sentiment leaning bullish ({avg:+.3f}) (+8)")
        elif label == "Bearish":
            score -= 8
            reasons.append(f"News sentiment leaning bearish ({avg:+.3f}) (-8)")
        else:
            reasons.append(f"News sentiment neutral ({avg:+.3f}) (no change)")

    if insider_signal:
        if insider_signal == "Bullish":
            score += 8
            reasons.append("Open-market insider buying detected (+8)")
        elif insider_signal == "Bearish":
            score -= 5
            reasons.append("Insider selling detected — may be routine (-5)")
        else:
            reasons.append("Insider activity mixed or neutral (no change)")

    return max(0, min(100, int(score))), reasons


# ── Signal labels ─────────────────────────────────────────────────────────────
def get_signal(score):
    if score >= 72:  return "Strong swing candidate"
    if score >= 58:  return "Potential swing setup — watch closely"
    if score >= 45:  return "Neutral — needs more confirmation"
    return "Weak setup — avoid or wait"

def get_signal_short(score):
    if score >= 72:  return "Strong"
    if score >= 58:  return "Watch"
    if score >= 45:  return "Neutral"
    return "Weak"
