
import pandas as pd

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

def add_indicators(df):
    df["MA20"]      = df["Close"].rolling(20).mean()
    df["MA50"]      = df["Close"].rolling(50).mean()
    df["RSI"]       = calculate_rsi(df)
    df["VolumeAvg"] = df["Volume"].rolling(20).mean()
    return df.dropna()

# ── Scoring ───────────────────────────────────────────────────────────────────
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
        score += 20
        reasons.append("Price above 50-day MA — historically favorable (+20)")
    else:
        score -= 20
        reasons.append("Price below 50-day MA — historically weaker setup (-20)")

    if ma20 > ma50:
        score += 15
        reasons.append("20-day MA above 50-day MA — bullish alignment signal (+15)")

    if 45 <= rsi <= 70:
        score += 15
        reasons.append(f"RSI {rsi:.1f} — momentum in healthy range (+15)")
    elif rsi > 75:
        score -= 15
        reasons.append(f"RSI {rsi:.1f} — potentially overbought (-15)")
    else:
        reasons.append(f"RSI {rsi:.1f} — outside ideal range, watch for direction (no change)")

    if vol > avg_vol:
        score += 10
        reasons.append("Volume above 20-day average — move may have conviction (+10)")
    else:
        reasons.append("Volume below average — move may lack conviction (no change)")

    if news_sentiment:
        label = news_sentiment.get("sentiment_label", "Neutral")
        avg   = news_sentiment.get("avg_score", 0)
        if label == "Bullish":
            score += 10
            reasons.append(f"News sentiment leaning bullish ({avg:+.3f}) (+10)")
        elif label == "Bearish":
            score -= 10
            reasons.append(f"News sentiment leaning bearish ({avg:+.3f}) (-10)")
        else:
            reasons.append(f"News sentiment neutral ({avg:+.3f}) (no change)")

    if insider_signal:
        if insider_signal == "Bullish":
            score += 10
            reasons.append("Open-market insider buying detected (+10)")
        elif insider_signal == "Bearish":
            score -= 5
            reasons.append("Insider selling detected — may be routine (-5)")
        else:
            reasons.append("Insider activity mixed or neutral (no change)")

    return max(0, min(100, int(score))), reasons

# ── Signal labels ─────────────────────────────────────────────────────────────
def get_signal(score):
    if score >= 70: return "Bullish signals align"
    if score >= 50: return "Neutral — watch carefully"
    return "Weak setup — exercise caution"

def get_signal_short(score):
    if score >= 70: return "Bullish"
    if score >= 50: return "Neutral"
    return "Weak"
