import requests

# ── Single ticker sentiment ───────────────────────────────────────────────────
def fetch_news_sentiment(ticker, alpha_key, increment_usage_fn):
    """
    Fetches news sentiment for a single ticker from Alpha Vantage.
    Costs 1 Alpha Vantage request.
    """
    if not alpha_key:
        return None, "Alpha Vantage key missing"

    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&tickers={ticker}"
            f"&limit=20"
            f"&apikey={alpha_key}"
        )
        increment_usage_fn("alpha_vantage", 1)
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if "Information" in data or "Note" in data:
            return None, data.get("Information") or data.get("Note")
        if "feed" not in data or not data["feed"]:
            return None, "No news found"

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
            if len(top_articles) < 5:
                top_articles.append({
                    "title":     article.get("title", "No title"),
                    "source":    article.get("source", "Unknown"),
                    "url":       article.get("url", ""),
                    "time":      article.get("time_published", "")[:8],
                    "sentiment": article.get("overall_sentiment_label", "Neutral"),
                    "summary":   article.get("summary", ""),
                })

        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        label = (
            "Bullish" if avg_score >= 0.15 else
            "Bearish" if avg_score <= -0.15 else
            "Neutral"
        )

        return {
            "article_count":   len(articles),
            "avg_score":       avg_score,
            "sentiment_label": label,
            "top_articles":    top_articles,
            "scored_articles": len(scores),
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ── Sector mapping ────────────────────────────────────────────────────────────
TICKER_SECTOR_MAP = {
    # AI & Machine Learning
    "NVDA": "AI & Machine Learning", "AMD": "AI & Machine Learning",
    "MSFT": "AI & Machine Learning", "GOOGL": "AI & Machine Learning",
    "META": "AI & Machine Learning", "AMZN": "AI & Machine Learning",
    "ORCL": "AI & Machine Learning", "IBM": "AI & Machine Learning",
    "PLTR": "AI & Machine Learning", "AI": "AI & Machine Learning",
    # Cybersecurity
    "CRWD": "Cybersecurity", "PANW": "Cybersecurity", "FTNT": "Cybersecurity",
    "ZS": "Cybersecurity", "S": "Cybersecurity", "OKTA": "Cybersecurity",
    "CYBR": "Cybersecurity", "TENB": "Cybersecurity", "RPD": "Cybersecurity",
    # Semiconductors
    "INTC": "Semiconductors", "QCOM": "Semiconductors", "AVGO": "Semiconductors",
    "MU": "Semiconductors", "AMAT": "Semiconductors", "LRCX": "Semiconductors",
    "KLAC": "Semiconductors", "TSM": "Semiconductors", "ASML": "Semiconductors",
    "MRVL": "Semiconductors", "ON": "Semiconductors", "SWKS": "Semiconductors",
    # Defense
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "GD": "Defense",
    "BA": "Defense", "HII": "Defense", "LDOS": "Defense", "CACI": "Defense",
    "SAIC": "Defense", "KTOS": "Defense", "AXON": "Defense", "BWXT": "Defense",
    # Biotech & Pharma
    "MRNA": "Biotech & Pharma", "BNTX": "Biotech & Pharma", "REGN": "Biotech & Pharma",
    "VRTX": "Biotech & Pharma", "BIIB": "Biotech & Pharma", "GILD": "Biotech & Pharma",
    "AMGN": "Biotech & Pharma", "ILMN": "Biotech & Pharma", "RARE": "Biotech & Pharma",
    "EXAS": "Biotech & Pharma", "LLY": "Biotech & Pharma", "PFE": "Biotech & Pharma",
    "JNJ": "Biotech & Pharma", "ABBV": "Biotech & Pharma", "BMY": "Biotech & Pharma",
    # Cloud & SaaS
    "CRM": "Cloud & SaaS", "NOW": "Cloud & SaaS", "SNOW": "Cloud & SaaS",
    "DDOG": "Cloud & SaaS", "MDB": "Cloud & SaaS", "NET": "Cloud & SaaS",
    "HUBS": "Cloud & SaaS", "ZM": "Cloud & SaaS", "TEAM": "Cloud & SaaS",
    "WDAY": "Cloud & SaaS", "VEEV": "Cloud & SaaS", "ZI": "Cloud & SaaS",
    # Mega-Cap Tech
    "AAPL": "Mega-Cap Tech", "NFLX": "Mega-Cap Tech", "ADBE": "Mega-Cap Tech",
    "TSLA": "Mega-Cap Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "SQ": "Financials", "COIN": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "OXY": "Energy", "PSX": "Energy", "MPC": "Energy", "EOG": "Energy",
    "PXD": "Energy", "HAL": "Energy",
    # Consumer & Retail
    "AMZN": "Consumer & Retail", "WMT": "Consumer & Retail", "TGT": "Consumer & Retail",
    "COST": "Consumer & Retail", "HD": "Consumer & Retail", "LOW": "Consumer & Retail",
    "NKE": "Consumer & Retail", "SBUX": "Consumer & Retail", "MCD": "Consumer & Retail",
    "DIS": "Consumer & Retail",
    # ETFs
    "SPY": "ETFs", "QQQ": "ETFs", "IWM": "ETFs", "DIA": "ETFs",
    "XLK": "ETFs", "XLF": "ETFs", "XLV": "ETFs", "GLD": "ETFs",
    "TLT": "ETFs", "VNQ": "ETFs", "ARKK": "ETFs", "SOXX": "ETFs",
}

def get_sector(ticker):
    return TICKER_SECTOR_MAP.get(ticker.upper(), "Other")


# ── Market-wide sentiment scan ────────────────────────────────────────────────
def fetch_market_sentiment_scan(alpha_key, increment_usage_fn, pages=3):
    """
    Fetches broad market news from Alpha Vantage without a ticker filter.
    Fetches multiple pages (each page = 1 request) to get more coverage.
    Returns top 20 most-mentioned tickers with sector labels.
    Also returns sector summary showing which sectors have the most activity.

    Args:
        alpha_key:          Alpha Vantage API key
        increment_usage_fn: callable — increment_usage from database module
        pages:              int — number of pages to fetch (each costs 1 request)
                            default 3 gives ~150 articles covering 1-2 days of news

    Returns:
        (result_dict, error_string) — error_string is None on success
        result_dict keys: tickers (top 20 list), sector_summary, total_articles,
                          pages_fetched
    """
    if not alpha_key:
        return None, "Alpha Vantage key missing"

    # Time offsets to get older articles on subsequent pages
    time_from_options = [
        None,           # page 1 — latest 50 articles
        "latest",       # page 2 — next batch
        "oldest",       # page 3 — go back further
    ]

    all_articles = []
    pages_fetched = 0

    for page_idx in range(pages):
        try:
            # Build URL — use sort variation to get different articles each page
            sort = "LATEST" if page_idx == 0 else "RELEVANCE" if page_idx == 1 else "EARLIEST"
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&limit=50"
                f"&sort={sort}"
                f"&apikey={alpha_key}"
            )
            increment_usage_fn("alpha_vantage", 1)
            resp = requests.get(url, timeout=15)
            data = resp.json()

            if "Information" in data or "Note" in data:
                # Hit rate limit — stop here, return what we have
                break
            if "feed" not in data or not data["feed"]:
                break

            all_articles.extend(data["feed"])
            pages_fetched += 1

        except requests.exceptions.Timeout:
            break
        except Exception:
            break

    if not all_articles:
        return None, "No market news returned"

    # Deduplicate articles by URL
    seen_urls   = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)

    # Aggregate ticker mentions and sentiment
    ticker_data = {}
    for article in unique_articles:
        pub_time = article.get("time_published", "")

        for ts in article.get("ticker_sentiment", []):
            t     = ts.get("ticker", "").upper()
            score = ts.get("ticker_sentiment_score", 0)

            # Skip crypto, forex, and malformed tickers
            if not t or len(t) > 5 or ":" in t or "." in t:
                continue

            try:
                score = float(score)
            except:
                score = 0.0

            if t not in ticker_data:
                ticker_data[t] = {
                    "ticker":        t,
                    "sector":        get_sector(t),
                    "mention_count": 0,
                    "score_sum":     0.0,
                    "scores":        [],
                    "top_headline":  article.get("title", ""),
                    "top_source":    article.get("source", ""),
                    "top_url":       article.get("url", ""),
                    "top_time":      pub_time[:8],
                    "latest_time":   pub_time,
                }
            else:
                # Keep the most recent article as top headline
                if pub_time > ticker_data[t]["latest_time"]:
                    ticker_data[t]["top_headline"] = article.get("title", "")
                    ticker_data[t]["top_source"]   = article.get("source", "")
                    ticker_data[t]["top_url"]       = article.get("url", "")
                    ticker_data[t]["top_time"]      = pub_time[:8]
                    ticker_data[t]["latest_time"]   = pub_time

            ticker_data[t]["mention_count"] += 1
            ticker_data[t]["score_sum"]     += score
            ticker_data[t]["scores"].append(score)

    # Build ticker results
    ticker_results = []
    for t, d in ticker_data.items():
        avg = round(d["score_sum"] / d["mention_count"], 4) if d["mention_count"] else 0.0
        sent_label = (
            "Bullish" if avg >= 0.15 else
            "Bearish" if avg <= -0.15 else
            "Neutral"
        )
        ticker_results.append({
            "Ticker":        t,
            "Sector":        d["sector"],
            "Mentions":      d["mention_count"],
            "Avg Sentiment": avg,
            "Sentiment":     sent_label,
            "Top Headline":  d["top_headline"],
            "Source":        d["top_source"],
            "URL":           d["top_url"],
            "Time":          d["top_time"],
        })

    # Sort by mentions then sentiment — take top 20
    ticker_results = sorted(
        ticker_results,
        key=lambda x: (x["Mentions"], x["Avg Sentiment"]),
        reverse=True
    )[:20]

    # Build sector summary
    sector_data = {}
    for t, d in ticker_data.items():
        sector = d["sector"]
        avg    = round(d["score_sum"] / d["mention_count"], 4) if d["mention_count"] else 0.0
        if sector not in sector_data:
            sector_data[sector] = {
                "sector":         sector,
                "ticker_count":   0,
                "total_mentions": 0,
                "score_sum":      0.0,
                "bullish_count":  0,
                "bearish_count":  0,
                "neutral_count":  0,
                "top_ticker":     t,
                "top_mentions":   d["mention_count"],
            }
        sector_data[sector]["ticker_count"]   += 1
        sector_data[sector]["total_mentions"] += d["mention_count"]
        sector_data[sector]["score_sum"]      += avg

        if avg >= 0.15:
            sector_data[sector]["bullish_count"] += 1
        elif avg <= -0.15:
            sector_data[sector]["bearish_count"] += 1
        else:
            sector_data[sector]["neutral_count"] += 1

        if d["mention_count"] > sector_data[sector]["top_mentions"]:
            sector_data[sector]["top_ticker"]   = t
            sector_data[sector]["top_mentions"] = d["mention_count"]

    sector_results = []
    for sector, d in sector_data.items():
        avg_sent = round(d["score_sum"] / d["ticker_count"], 4) if d["ticker_count"] else 0.0
        sent_label = (
            "Bullish" if avg_sent >= 0.15 else
            "Bearish" if avg_sent <= -0.15 else
            "Neutral"
        )
        sector_results.append({
            "Sector":          sector,
            "Total Mentions":  d["total_mentions"],
            "Tickers Covered": d["ticker_count"],
            "Avg Sentiment":   avg_sent,
            "Sentiment":       sent_label,
            "Bullish Tickers": d["bullish_count"],
            "Bearish Tickers": d["bearish_count"],
            "Top Ticker":      d["top_ticker"],
        })

    sector_results = sorted(
        sector_results,
        key=lambda x: x["Total Mentions"],
        reverse=True
    )

    return {
        "tickers":        ticker_results,
        "sector_summary": sector_results,
        "total_articles": len(unique_articles),
        "pages_fetched":  pages_fetched,
    }, None
