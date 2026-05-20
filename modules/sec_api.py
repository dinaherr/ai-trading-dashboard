import requests
from datetime import date

# ── Constants ─────────────────────────────────────────────────────────────────
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# ── Header builder ────────────────────────────────────────────────────────────
def get_sec_headers(sec_agent):
    """
    Builds SEC request headers from the User-Agent secret.
    Never displayed in the UI.
    """
    return {
        "User-Agent":      sec_agent,
        "Accept-Encoding": "gzip, deflate",
    }

# ── Company lookup ────────────────────────────────────────────────────────────
def get_sec_company_info(ticker, sec_agent, increment_usage_fn,
                          save_cache_fn, load_cache_fn):
    """
    Maps ticker → CIK using SEC's public company_tickers.json.
    Checks SQLite cache first. Costs 1 SEC request on first lookup only.
    """
    ticker = ticker.upper().strip()

    cached = load_cache_fn(ticker)
    if cached:
        cached["from_cache"] = True
        return cached, None

    try:
        increment_usage_fn("sec", 1)
        headers = get_sec_headers(sec_agent)
        resp    = requests.get(SEC_TICKERS_URL, headers=headers, timeout=10)

        if resp.status_code != 200:
            return None, f"SEC returned status {resp.status_code}"

        data    = resp.json()
        cik     = None
        company = None

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik     = str(entry["cik_str"]).zfill(10)
                company = entry.get("title", "Unknown")
                break

        if not cik:
            return None, f"Ticker {ticker} not found in SEC database"

        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        result = {
            "cik":             cik,
            "company_name":    company,
            "submissions_url": submissions_url,
            "fetched_date":    date.today().isoformat(),
            "from_cache":      False,
        }
        save_cache_fn(ticker, cik, company, submissions_url)
        return result, None

    except requests.exceptions.Timeout:
        return None, "SEC request timed out — try again"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ── Filing retrieval ──────────────────────────────────────────────────────────
def get_sec_filings(ticker, cik, sec_agent, increment_usage_fn,
                     save_cache_fn, load_cache_fn, form_types=None):
    """
    Fetches recent SEC filings for a given CIK.
    Checks SQLite cache first. Costs 1 SEC request on first fetch per day only.
    """
    if form_types is None:
        form_types = ["4", "10-Q", "10-K", "8-K", "13F-HR", "13F-HR/A"]

    cached = load_cache_fn(ticker)
    if cached:
        return cached, None, True

    try:
        increment_usage_fn("sec", 1)
        url     = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = get_sec_headers(sec_agent)
        resp    = requests.get(url, headers=headers, timeout=15)

        if resp.status_code != 200:
            return None, f"SEC returned status {resp.status_code}", False

        data    = resp.json()
        recent  = data.get("filings", {}).get("recent", {})
        forms   = recent.get("form", [])
        dates   = recent.get("filingDate", [])
        reports = recent.get("reportDate", [])
        accnos  = recent.get("accessionNumber", [])
        docs    = recent.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form not in form_types:
                continue

            accno_clean = accnos[i].replace("-", "") if i < len(accnos) else ""

            if i < len(docs) and docs[i] and accno_clean:
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accno_clean}/{docs[i]}"
                )
            else:
                filing_url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar"
                    f"?action=getcompany&CIK={cik}"
                    f"&type={form}&dateb=&owner=include&count=10"
                )

            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accno_clean}/"
                if accno_clean else ""
            )

            filings.append({
                "Form":        form,
                "Filed":       dates[i]   if i < len(dates)   else "",
                "Report Date": reports[i] if i < len(reports) else "",
                "Accession":   accnos[i]  if i < len(accnos)  else "",
                "Filing URL":  filing_url,
                "Index URL":   index_url,
            })

            if len(filings) >= 50:
                break

        save_cache_fn(ticker, filings)
        return filings, None, False

    except requests.exceptions.Timeout:
        return None, "SEC request timed out — try again", False
    except Exception as e:
        return None, f"Error: {str(e)}", False
