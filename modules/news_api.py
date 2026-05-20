import requests

# ── Single ticker sentiment ───────────────────────────────────────────────────
def fetch_news_sentiment(ticker, alpha_key, increment_usage_fn):
    """
    Fetches news sentiment for a single ticker from Alpha Vantage.
    Costs 1 Alpha Vantage request.

    Args:
        ticker:             stock ticker symbol
        alpha_key:          Alpha Vantage API key
        increment_usage_fn: callable — increment_usage from database module

    Returns:
        (result_dict, error_string) — error_string is None on success
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


# ── Market-wide sentiment scan ────────────────────────────────────────────────
def fetch_market_sentiment_scan(alpha_key, increment_usage_fn):
    """
    Fetches broad market news without a ticker filter.
    Costs exactly 1 Alpha Vantage request.
    Returns top 15 most-mentioned tickers ranked by mentions and sentiment.

    Args:
        alpha_key:          Alpha Vantage API key
        increment_usage_fn: callable — increment_usage from database module

    Returns:
        (results_list, error_string) — error_string is None on success
    """
    if not alpha_key:
        return None, "Alpha Vantage key missing"

    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT"
            f"&limit=50"
            f"&sort=LATEST"
            f"&apikey={alpha_key}"
        )
        increment_usage_fn("alpha_vantage", 1)
        resp = requests.get(url, timeout=15)
        data = resp.json()

        if "Information" in data or "Note" in data:
            return None, data.get("Information") or data.get("Note")
        if "feed" not in data or not data["feed"]:
            return None, "No market news returned"

        articles    = data["feed"]
        ticker_data = {}

        for article in articles:
            for ts in article.get("ticker_sentiment", []):
                t     = ts.get("ticker", "").upper()
                score = ts.get("ticker_sentiment_score", 0)

                if not t or len(t) > 5 or ":" in t or "." in t:
                    continue

                try:
                    score = float(score)
                except:
                    score = 0.0

                if t not in ticker_data:
                    ticker_data[t] = {
                        "ticker":        t,
                        "mention_count": 0,
                        "score_sum":     0.0,
                        "scores":        [],
                        "top_headline":  article.get("title", ""),
                        "top_source":    article.get("source", ""),
                        "top_url":       article.get("url", ""),
                        "top_time":      article.get("time_published", "")[:8],
                    }

                ticker_data[t]["mention_count"] += 1
                ticker_data[t]["score_sum"]     += score
                ticker_data[t]["scores"].append(score)

        results = []
        for t, d in ticker_data.items():
            avg = round(d["score_sum"] / d["mention_count"], 4) if d["mention_count"] else 0.0
            sent_label = (
                "Bullish" if avg >= 0.15 else
                "Bearish" if avg <= -0.15 else
                "Neutral"
            )
            results.append({
                "Ticker":        t,
                "Mentions":      d["mention_count"],
                "Avg Sentiment": avg,
                "Sentiment":     sent_label,
                "Top Headline":  d["top_headline"],
                "Source":        d["top_source"],
                "URL":           d["top_url"],
                "Time":          d["top_time"],
            })

        results = sorted(
            results,
            key=lambda x: (x["Mentions"], x["Avg Sentiment"]),
            reverse=True
        )[:15]

        return results, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"
