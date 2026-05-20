# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI research assistant helping with paper trading education. "
    "Never say buy or sell. Never predict prices. Use hedged language: "
    "'signals suggest', 'may indicate', 'historically'. "
    "End every response with: This analysis is for paper trading and "
    "educational research only. Not financial advice."
)

# ── Internal helper ───────────────────────────────────────────────────────────
def _safe_float(val):
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    return float(val)

# ── ChatGPT copy-paste prompt ─────────────────────────────────────────────────
def build_chatgpt_prompt(ticker, latest, score, reasons,
                          sentiment_data=None, insider_data=None):
    """
    Builds a fully formatted prompt the user can paste into chatgpt.com for free.
    No API call is made here.
    """
    lines = [
        "You are an AI stock research assistant for paper trading and educational purposes only.",
        f"Analyze the following publicly available data for {ticker}.",
        "",
        "Return clearly labeled sections:",
        "1. Bullish factors",
        "2. Bearish risks",
        "3. Neutral or unclear points",
        "4. Possible near-term market impact to watch",
        "5. What a paper trader might monitor next",
        "",
        "Rules: Do not say buy or sell. Do not predict prices. Use hedged language.",
        "End with: This analysis is for paper trading and educational research only. "
        "Not financial advice.",
        "",
        f"--- TECHNICAL DATA FOR {ticker} ---",
        f"Close: ${_safe_float(latest['Close']):.2f}",
        f"RSI: {_safe_float(latest['RSI']):.1f}",
        f"MA20: ${_safe_float(latest['MA20']):.2f}",
        f"MA50: ${_safe_float(latest['MA50']):.2f}",
        f"Volume: {_safe_float(latest['Volume']):,.0f}",
        f"Research Score: {score}/100 (educational metric only)",
        "",
        "Score signals:",
    ]

    for r in reasons:
        lines.append(f"  - {r}")

    if sentiment_data:
        lines += [
            "",
            "--- PUBLIC NEWS SENTIMENT ---",
            f"Label: {sentiment_data['sentiment_label']}",
            f"Avg Score: {sentiment_data['avg_score']}",
            f"Articles: {sentiment_data['article_count']}",
            "Headlines:",
        ]
        for a in sentiment_data["top_articles"]:
            lines.append(f"  - [{a['sentiment']}] {a['title']} ({a['source']})")
            if a.get("summary"):
                lines.append(f"    {a['summary'][:200]}...")

    if insider_data:
        lines += [
            "",
            "--- PUBLIC INSIDER DISCLOSURES (SEC filings, last 90 days) ---",
            f"Signal: {insider_data['insider_signal']}",
            f"Open-market buys: {insider_data.get('open_buy_count', insider_data['buy_count'])}",
            f"Open-market sells: {insider_data.get('open_sell_count', insider_data['sell_count'])}",
            f"Net shares: {insider_data['net_shares']:,}",
            "Recent:",
        ]
        for t in insider_data["transactions"][:5]:
            lines.append(
                f"  - {t['Date']} | {t['Name']} | {t['Type']} "
                f"| {t['Shares']} @ {t['Price']}"
            )

    lines += [
        "",
        "---",
        "For paper trading and educational research only. Not financial advice.",
    ]

    return "\n".join(lines)


# ── OpenAI full analysis ──────────────────────────────────────────────────────
def generate_ai_analysis(client, ticker, latest, score,
                          sentiment_data=None, insider_data=None):
    """
    Calls OpenAI API to generate a research summary.
    client is passed in — not a global.
    """
    s_block = ""
    if sentiment_data:
        headlines = "\n".join([
            f"  - [{a['sentiment']}] {a['title']} — {a.get('summary', '')[:150]}"
            for a in sentiment_data.get("top_articles", [])
        ])
        s_block = (
            f"\nNews Sentiment: {sentiment_data['sentiment_label']} "
            f"(score {sentiment_data['avg_score']})\n"
            f"Headlines:\n{headlines}\n"
        )

    i_block = ""
    if insider_data:
        i_block = (
            f"\nPublic Insider Disclosures (90 days): {insider_data['insider_signal']}\n"
            f"Open-mkt buys: "
            f"{insider_data.get('open_buy_count', insider_data['buy_count'])} | "
            f"Open-mkt sells: "
            f"{insider_data.get('open_sell_count', insider_data['sell_count'])}\n"
            f"Net shares: {insider_data['net_shares']:,}\n"
        )

    prompt = (
        f"Analyze for paper trading research: {ticker}\n"
        f"Close: ${_safe_float(latest['Close']):.2f} | "
        f"RSI: {_safe_float(latest['RSI']):.1f}\n"
        f"MA20: ${_safe_float(latest['MA20']):.2f} | "
        f"MA50: ${_safe_float(latest['MA50']):.2f}\n"
        f"Volume: {_safe_float(latest['Volume']):.0f} | "
        f"Score: {score}/100\n"
        f"{s_block}{i_block}\n"
        f"Return: 1) Technical setup 2) MA trend 3) RSI reading "
        f"4) Volume conviction 5) News signals 6) Insider signals "
        f"7) One thing to watch. Hedged language throughout."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content


# ── OpenAI article analysis ───────────────────────────────────────────────────
def analyze_articles_with_ai(client, ticker, articles):
    """
    Sends article headlines and summaries to OpenAI for bullish/bearish breakdown.
    client is passed in — not a global.
    """
    article_text = "\n".join([
        f"Title: {a['title']}\n"
        f"Source: {a['source']} | {a['sentiment']}\n"
        f"Summary: {a.get('summary', '')[:300]}\n"
        f"URL: {a['url']}"
        for a in articles
    ])

    prompt = (
        f"Analyze these public news articles for {ticker} "
        f"(paper trading research only).\n"
        f"{article_text}\n"
        f"Return: 1) Bullish factors 2) Bearish risks 3) Neutral points "
        f"4) Possible market impact 5) What to watch next. "
        f"No buy/sell. Hedged language."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content
