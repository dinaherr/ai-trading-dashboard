import requests
from datetime import datetime, timedelta

# ── Insider transactions ──────────────────────────────────────────────────────
def fetch_insider_transactions(ticker, finnhub_key, increment_usage_fn):
    """
    Fetches recent insider transactions from Finnhub.
    Filters to last 90 days. Weights open-market buys more than grants.
    Costs 1 Finnhub request.

    Args:
        ticker:             stock ticker symbol
        finnhub_key:        Finnhub API key
        increment_usage_fn: callable — increment_usage from database module

    Returns:
        (result_dict, error_string) — error_string is None on success
    """
    if not finnhub_key:
        return None, "Finnhub key missing"

    try:
        url = (
            f"https://finnhub.io/api/v1/stock/insider-transactions"
            f"?symbol={ticker}&token={finnhub_key}"
        )
        increment_usage_fn("finnhub", 1)
        resp = requests.get(url, timeout=10)
        data = resp.json()

        transactions = data.get("data", [])
        if not transactions:
            return None, "No insider transactions found"

        cutoff = (datetime.now() - timedelta(days=90)).date()
        recent = []
        for t in transactions:
            try:
                tx_date = datetime.strptime(
                    t.get("transactionDate", ""), "%Y-%m-%d"
                ).date()
                if tx_date >= cutoff:
                    recent.append(t)
            except:
                pass

        if not recent:
            return None, "No transactions in last 90 days"

        open_buys  = [t for t in recent if t.get("transactionCode") == "P"]
        all_buys   = [t for t in recent if t.get("transactionCode") in ["P", "A"]]
        open_sells = [t for t in recent if t.get("transactionCode") == "S"]
        all_sells  = [t for t in recent if t.get("transactionCode") in ["S", "D"]]

        net_shares = (
            sum(t.get("share", 0) or 0 for t in all_buys) -
            sum(t.get("share", 0) or 0 for t in all_sells)
        )

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
                "Type":   (
                    "Open-Market Buy"   if code == "P" else
                    "Grant/Award"       if code == "A" else
                    "Open-Market Sell"  if code == "S" else
                    "Planned/Auto Sell"
                ),
                "Shares": f"{t.get('share', 0):,}",
                "Price":  f"${t.get('price', 0):.2f}" if t.get("price") else "N/A",
                "Value":  f"${(t.get('share', 0) or 0) * (t.get('price', 0) or 0):,.0f}",
            })

        return {
            "buy_count":       len(all_buys),
            "sell_count":      len(all_sells),
            "open_buy_count":  len(open_buys),
            "open_sell_count": len(open_sells),
            "net_shares":      net_shares,
            "insider_signal":  insider_signal,
            "recent_count":    len(recent),
            "transactions":    table_rows,
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"
