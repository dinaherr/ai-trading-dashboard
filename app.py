import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from datetime import datetime, date
import os

# ── Modules ───────────────────────────────────────────────────────────────────
from modules.database import (
    AV_DAILY_LIMIT,
    FINNHUB_DAILY_LIMIT,
    OPENAI_DAILY_LIMIT,
    SEC_DAILY_LIMIT,
    init_db,
    load_trades,
    save_trade,
    close_trade,
    delete_trade,
    get_usage_today,
    increment_usage,
    requests_remaining,
    save_news_cache,
    load_news_cache,
    save_insider_cache,
    load_insider_cache,
    save_signal_snapshot,
    load_signal_history,
    save_sec_company_cache,
    load_sec_company_cache,
    save_sec_filings_cache,
    load_sec_filings_cache,
)
from modules.scoring import (
    safe_float,
    add_indicators,
    score_stock,
    get_signal,
    get_signal_short,
)
from modules.market_data import (
    get_data,
    get_current_price,
)
from modules.news_api import (
    fetch_news_sentiment,
    fetch_market_sentiment_scan,
)
from modules.ai_analysis import (
    build_chatgpt_prompt,
    generate_ai_analysis,
    analyze_articles_with_ai,
)
from modules.sec_api import (
    get_sec_company_info,
    get_sec_filings,
)
from modules.ui_helpers import (
    render_request_gate,
    render_openai_gate,
    render_signal_banner,
    render_sentiment_banner,
    render_insider_banner,
    render_api_budget_sidebar,
    render_score_breakdown,
    render_snapshot_status,
    render_disclaimer,
)
from modules.discovery import (
    get_category_names,
    get_category_tickers,
    build_display_df,
    get_quick_stats,
)
from modules.insider_api import fetch_insider_transactions

# ── Bootstrap ─────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Trading Research Dashboard", layout="wide")
st.title("AI Trading Research Dashboard")
st.caption(
    "📊 An educational paper-trading research tool. "
    "All signals, scores, and AI summaries are for research and educational purposes only. "
    "Nothing here is financial advice. Past patterns do not guarantee future results."
)
st.info(
    "⚠️ **Disclaimer:** This dashboard is for personal research and paper-trading education only. "
    "It does not provide financial advice, investment recommendations, or trading signals. "
    "All data shown may be delayed, incomplete, or inaccurate. "
    "Never make real financial decisions based solely on this tool. "
    "Consult a licensed financial professional before investing."
)

# ── Constants ─────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "*For educational and paper-trading research only. "
    "Not financial advice. Signals may be delayed or inaccurate. "
    "Never make real investment decisions based solely on this tool.*"
)

# ── Secrets ───────────────────────────────────────────────────────────────────
openai_key  = st.secrets.get("OPENAI_API_KEY", None)
alpha_key   = st.secrets.get("ALPHA_VANTAGE_API_KEY", None)
finnhub_key = st.secrets.get("FINNHUB_API_KEY", None)
sec_agent   = st.secrets.get("SEC_USER_AGENT", "MyApp myemail@email.com")
client      = OpenAI(api_key=openai_key) if openai_key else None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Watchlist")
    tickers_input = st.text_input(
        "Enter tickers (comma separated)", "NVDA, CRWD, PANW, AMD, SPY"
    )
    period   = st.selectbox("Time period", ["3mo", "6mo", "1y", "2y"], index=1)
    selected = st.selectbox(
        "Deep-dive ticker",
        [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    )

    st.divider()
    render_api_budget_sidebar(
        [
            ("alpha_vantage", "Alpha Vantage", AV_DAILY_LIMIT),
            ("finnhub",       "Finnhub",       FINNHUB_DAILY_LIMIT),
            ("openai",        "OpenAI calls",  OPENAI_DAILY_LIMIT),
            ("sec",           "SEC EDGAR",     SEC_DAILY_LIMIT),
        ],
        get_usage_today,
        requests_remaining,
    )

    st.divider()
    st.header("Build Status")
    st.caption("✅ Phase 1 — Watchlist + charts + scoring")
    st.caption("✅ Phase 2 — News sentiment + insider trades")
    st.caption("✅ Phase 3 — Signal history + backtesting + scanner")
    st.caption("✅ Phase 4 — SEC company lookup + filings")
    st.caption("✅ Phase 4 — Market sentiment scanner")
    st.caption("🔒 Phase 4 next — 13F institutional tracker")
    st.caption("🔒 Phase 4 next — Politician disclosure tracker")
    st.caption("🔒 Phase 4 next — Catalyst calendar")

    st.divider()
    st.caption(
        "📋 For educational and paper-trading research only. "
        "Not financial advice. Consult a licensed financial professional "
        "before making any investment decisions."
    )

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# ── Cached market scan wrapper ────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def cached_market_sentiment_scan(key, pages):
    return fetch_market_sentiment_scan(key, increment_usage, pages=pages)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Watchlist Overview",
    "Deep Dive + Phase 2",
    "Backtesting",
    "Stock Discovery",
    "Market Sentiment Scanner",
    "Public Disclosures",
    "Paper Trade Log",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Watchlist Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Watchlist Summary")
    st.caption(
        "Research scores are educational metrics only. "
        "Higher score = more bullish technical alignment. Not a trade recommendation."
    )

    summary_rows = []
    all_data     = {}

    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty: continue
        data = add_indicators(raw.copy())
        if data.empty: continue
        all_data[ticker] = data
        latest   = data.iloc[-1]
        score, _ = score_stock(data)
        summary_rows.append({
            "Ticker":           ticker,
            "Close":            f"${safe_float(latest['Close']):.2f}",
            "RSI":              f"{safe_float(latest['RSI']):.1f}",
            "MA20":             f"${safe_float(latest['MA20']):.2f}",
            "MA50":             f"${safe_float(latest['MA50']):.2f}",
            "Research Score":   f"{score}/100",
            "Signal Alignment": get_signal_short(score),
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf["_s"] = sdf["Research Score"].str.replace("/100", "").astype(int)
        sdf = sdf.sort_values("_s", ascending=False).drop(columns=["_s"])
        st.dataframe(sdf, use_container_width=True, hide_index=True)
    render_disclaimer(DISCLAIMER)
    st.divider()

    for ticker in tickers:
        if ticker not in all_data:
            st.error(f"No data for {ticker}"); continue

        data   = all_data[ticker]
        latest = data.iloc[-1]
        score, reasons = score_stock(data)
        signal = get_signal(score)

        st.subheader(ticker)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close",          f"${safe_float(latest['Close']):.2f}")
        c2.metric("RSI",            f"{safe_float(latest['RSI']):.1f}")
        c3.metric("Volume",         f"{safe_float(latest['Volume']):,.0f}")
        c4.metric("Research Score", f"{score}/100")
        render_signal_banner(score, signal)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
        fig.update_layout(
            height=400, xaxis_rangeslider_visible=False,
            title=f"{ticker} — {period} (data may be delayed ~15 min)"
        )
        st.plotly_chart(fig, use_container_width=True)

        render_score_breakdown(reasons, DISCLAIMER)

        with st.expander("AI Research Summary (uses OpenAI API)"):
            if client is None:
                st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
            else:
                st.caption("AI summaries are for research only.")
                if render_openai_gate(
                    f"ai_{ticker}",
                    f"Generate summary for {ticker}",
                    OPENAI_DAILY_LIMIT,
                    get_usage_today,
                    requests_remaining,
                ):
                    increment_usage("openai", 1)
                    with st.spinner(f"Generating summary for {ticker}..."):
                        st.write(generate_ai_analysis(client, ticker, latest, score))
                    render_disclaimer(DISCLAIMER)

        st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Dive + Phase 2
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Phase 2 Signal Alignment Summary")
    st.caption(
        "All data from public APIs. Insider data = public SEC filings only. For research only."
    )

    p2_rows = []
    for ticker in tickers:
        raw = get_data(ticker, period)
        if raw.empty: continue
        data = add_indicators(raw.copy())
        if data.empty: continue
        cs  = load_news_cache(ticker)    or st.session_state.get(f"sentiment_{ticker}")
        ci  = load_insider_cache(ticker) or st.session_state.get(f"insider_{ticker}")
        sl  = cs["sentiment_label"] if cs else "Not fetched"
        is_ = ci["insider_signal"]  if ci else "Not fetched"
        ts, _ = score_stock(data)
        comb, _ = score_stock(
            data, news_sentiment=cs,
            insider_signal=is_ if is_ not in ["Not fetched", "—"] else None
        )
        p2_rows.append({
            "Ticker":             ticker,
            "Tech Score":         f"{ts}/100",
            "News Sentiment":     sl,
            "Insider Disclosure": is_,
            "Combined Score":     f"{comb}/100",
            "Signal Alignment":   get_signal_short(comb),
        })

    if p2_rows:
        p2df = pd.DataFrame(p2_rows)
        p2df["_s"] = p2df["Combined Score"].str.replace("/100", "").astype(int)
        p2df = p2df.sort_values("_s", ascending=False).drop(columns=["_s"])
        st.dataframe(p2df, use_container_width=True, hide_index=True)
    render_disclaimer(DISCLAIMER)
    st.divider()

    st.subheader(f"Deep Dive: {selected}")
    raw = get_data(selected, period)

    if raw.empty:
        st.error(f"No data for {selected}")
    else:
        data   = add_indicators(raw.copy())
        latest = data.iloc[-1]
        cs     = load_news_cache(selected)    or st.session_state.get(f"sentiment_{selected}")
        ci     = load_insider_cache(selected) or st.session_state.get(f"insider_{selected}")
        isig   = ci["insider_signal"] if ci else None
        score, reasons = score_stock(data, news_sentiment=cs, insider_signal=isig)
        signal = get_signal(score)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Research Score",   f"{score}/100")
        c2.metric("Signal Alignment", get_signal_short(score))
        c3.metric("RSI",              f"{safe_float(latest['RSI']):.1f}")
        c4.metric("Close",            f"${safe_float(latest['Close']):.2f}")
        render_signal_banner(score, signal)

        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price"
        ))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
        fig2.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
        fig2.update_layout(
            height=500, xaxis_rangeslider_visible=False,
            title=f"{selected} — Deep Dive (data may be delayed ~15 min)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        render_score_breakdown(reasons, DISCLAIMER)

        snap_result = save_signal_snapshot(
            ticker               = selected,
            close                = safe_float(latest["Close"]),
            rsi                  = safe_float(latest["RSI"]),
            ma20                 = safe_float(latest["MA20"]),
            ma50                 = safe_float(latest["MA50"]),
            volume               = safe_float(latest["Volume"]),
            technical_score      = score_stock(data)[0],
            news_sentiment_label = cs["sentiment_label"] if cs else None,
            insider_signal       = isig,
            combined_score       = score,
            final_signal         = get_signal_short(score),
        )
        render_snapshot_status(snap_result, selected)

        # ── News sentiment ────────────────────────────────────────────────────
        with st.expander("News Sentiment — Alpha Vantage (public news data)", expanded=False):
            st.caption("Derived from publicly available articles. Does not predict price movement.")
            if not alpha_key:
                st.warning("Add ALPHA_VANTAGE_API_KEY to Streamlit Secrets.")
            else:
                if render_request_gate(
                    "alpha_vantage", AV_DAILY_LIMIT, selected,
                    "Alpha Vantage", f"av_{selected}",
                    get_usage_today, requests_remaining,
                ):
                    with st.spinner(f"Fetching news sentiment for {selected}..."):
                        result, err = fetch_news_sentiment(
                            selected, alpha_key, increment_usage
                        )
                    if err:
                        st.error(f"Could not load sentiment: {err}")
                    else:
                        save_news_cache(selected, result)
                        st.session_state[f"sentiment_{selected}"] = result
                        st.rerun()

                cached = (
                    load_news_cache(selected) or
                    st.session_state.get(f"sentiment_{selected}")
                )
                if cached:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Sentiment Lean",  cached["sentiment_label"])
                    s2.metric("Avg Score",       cached["avg_score"])
                    s3.metric("Articles Found",  cached["article_count"])
                    render_sentiment_banner(cached["sentiment_label"])

                    st.markdown("**Top recent public headlines:**")
                    for a in cached["top_articles"][:3]:
                        t = a["time"]
                        if len(t) == 8:
                            t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
                        st.markdown(
                            f"- [{a['title']}]({a['url']})  \n"
                            f"  *{a['source']} · {t} · {a['sentiment']}*"
                        )

                    st.divider()
                    if client:
                        st.markdown("**Analyze articles with AI**")
                        if render_openai_gate(
                            f"analyze_articles_{selected}",
                            "Analyze News Articles",
                            OPENAI_DAILY_LIMIT,
                            get_usage_today,
                            requests_remaining,
                        ):
                            increment_usage("openai", 1)
                            with st.spinner("Analyzing articles..."):
                                st.write(analyze_articles_with_ai(
                                    client, selected, cached["top_articles"]
                                ))
                            render_disclaimer(DISCLAIMER)

                    st.divider()
                    st.markdown("**Use ChatGPT instead — free**")
                    st.caption("Copy → paste into chatgpt.com. No API cost.")
                    st.code(build_chatgpt_prompt(
                        selected, latest, score, reasons,
                        sentiment_data=cached,
                        insider_data=(
                            load_insider_cache(selected) or
                            st.session_state.get(f"insider_{selected}")
                        ),
                    ), language="")
                else:
                    st.caption("No sentiment data yet. Confirm above to fetch.")

        # ── Insider activity ──────────────────────────────────────────────────
        with st.expander(
            "Insider Disclosure Activity — Finnhub (public SEC filings only)",
            expanded=False
        ):
            st.caption(
                "All data from public regulatory disclosures (SEC Form 4). "
                "Selling is often routine. Open-market buying is generally a stronger signal."
            )
            if not finnhub_key:
                st.warning("Add FINNHUB_API_KEY to Streamlit Secrets.")
            else:
                if render_request_gate(
                    "finnhub", FINNHUB_DAILY_LIMIT, selected,
                    "Finnhub", f"fh_{selected}",
                    get_usage_today, requests_remaining,
                ):
                    with st.spinner(f"Fetching insider disclosures for {selected}..."):
                        ir, ie = fetch_insider_transactions(
                            selected, finnhub_key, increment_usage
                        )
                    if ie:
                        st.error(f"Could not load insider data: {ie}")
                    else:
                        save_insider_cache(selected, ir)
                        st.session_state[f"insider_{selected}"] = ir
                        st.rerun()

                ci2 = (
                    load_insider_cache(selected) or
                    st.session_state.get(f"insider_{selected}")
                )
                if ci2:
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Disclosure Signal", ci2["insider_signal"])
                    i2.metric("Open-Mkt Buys",     ci2.get("open_buy_count",  ci2["buy_count"]))
                    i3.metric("Open-Mkt Sells",     ci2.get("open_sell_count", ci2["sell_count"]))
                    render_insider_banner(ci2["insider_signal"], ci2["net_shares"])
                    st.caption(
                        "Open-Market Buy = strongest signal. "
                        "Grant = compensation. Planned Sell = often pre-scheduled."
                    )
                    if ci2["transactions"]:
                        st.dataframe(
                            pd.DataFrame(ci2["transactions"]),
                            use_container_width=True, hide_index=True
                        )
                    render_disclaimer(DISCLAIMER)
                else:
                    st.caption("No insider data yet. Confirm above to fetch.")

        # ── AI Summary ────────────────────────────────────────────────────────
        st.subheader("AI Research Summary")
        st.caption("AI summaries are for paper trading research only.")
        st.markdown("**Option 1 — Free: Copy prompt → paste into chatgpt.com**")
        st.code(build_chatgpt_prompt(
            selected, latest, score, reasons,
            sentiment_data=cs, insider_data=ci,
        ), language="")

        st.markdown("**Option 2 — Use OpenAI API (costs tokens)**")
        if client is None:
            st.warning("Add OPENAI_API_KEY to Streamlit Secrets.")
        elif render_openai_gate(
            "ai_deepdive",
            f"Generate AI summary for {selected}",
            OPENAI_DAILY_LIMIT,
            get_usage_today,
            requests_remaining,
        ):
            increment_usage("openai", 1)
            with st.spinner("Generating research summary..."):
                st.info(generate_ai_analysis(
                    client, selected, latest, score,
                    sentiment_data=cs, insider_data=ci,
                ))
            render_disclaimer(DISCLAIMER)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Backtesting
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Signal Backtesting")
    st.info(
        "⚠️ **Backtesting disclaimer:** Past signal performance does not guarantee future results. "
        "Returns shown are hypothetical and based on signal snapshots vs current price. "
        "For educational research only. Not financial advice."
    )

    history_df = load_signal_history()

    if history_df.empty:
        st.info("No signal history yet. Visit the Deep Dive tab to start saving daily snapshots.")
    else:
        rows = []
        for _, row in history_df.iterrows():
            current = get_current_price(row["ticker"])
            if current and row["close"] > 0:
                ret_pct    = ((current - row["close"]) / row["close"]) * 100
                profitable = ret_pct > 0
            else:
                ret_pct = None; profitable = None
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

        bt_df = pd.DataFrame(rows)
        valid = bt_df.dropna(subset=["Return %"])

        st.subheader("Overall Backtest Summary")
        st.caption("Hypothetical returns based on snapshot close vs current price. Educational only.")

        if not valid.empty:
            win_rate   = valid["Profitable"].mean() * 100
            avg_return = valid["Return %"].mean()
            best       = valid.loc[valid["Return %"].idxmax()]
            worst      = valid.loc[valid["Return %"].idxmin()]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate",     f"{win_rate:.1f}%")
            m2.metric("Avg Return",   f"{avg_return:+.2f}%")
            m3.metric("Best Signal",  f"{best['Ticker']} {best['Return %']:+.2f}%")
            m4.metric("Worst Signal", f"{worst['Ticker']} {worst['Return %']:+.2f}%")
            sig_avg = (
                valid.groupby("Signal")["Return %"]
                .agg(["mean", "count"]).reset_index()
                .rename(columns={"mean": "Avg Return %", "count": "# Signals"})
            )
            sig_avg["Avg Return %"] = sig_avg["Avg Return %"].round(2)
            st.dataframe(sig_avg, use_container_width=True, hide_index=True)

        display_df = bt_df.copy()
        display_df["Return %"]      = display_df["Return %"].apply(
            lambda x: f"{x:+.2f}%" if x is not None else "Pending"
        )
        display_df["Current Price"] = display_df["Current Price"].apply(
            lambda x: f"${x:.2f}" if x else "N/A"
        )
        display_df["Close@Snap"] = display_df["Close@Snap"].apply(lambda x: f"${x:.2f}")
        st.dataframe(
            display_df.drop(columns=["Profitable"]),
            use_container_width=True, hide_index=True
        )
        render_disclaimer(DISCLAIMER)

        if not valid.empty:
            st.divider()
            st.subheader("Backtest Charts")
            st.caption("Educational only. Based on limited snapshots — interpret carefully.")

            st.markdown("**Average return by research score range**")
            st.caption("Do higher research scores correlate with better hypothetical returns?")
            valid2 = valid.copy()
            valid2["Score Range"] = pd.cut(
                valid2["Score@Snap"], bins=[0, 40, 55, 70, 100],
                labels=["0–40 (Weak)", "41–55 (Neutral-Low)",
                        "56–70 (Neutral-High)", "71–100 (Bullish)"]
            )
            sr_avg = valid2.groupby(
                "Score Range", observed=True
            )["Return %"].mean().reset_index()
            fig_sr = px.bar(
                sr_avg, x="Score Range", y="Return %",
                title="Avg Hypothetical Return by Score Range",
                color="Return %", color_continuous_scale="RdYlGn"
            )
            fig_sr.update_layout(height=350)
            st.plotly_chart(fig_sr, use_container_width=True)

            st.markdown("**Win rate by signal type**")
            st.caption("What percentage of each signal type resulted in a positive return?")
            wr_sig = (
                valid.groupby("Signal")["Profitable"]
                .agg(lambda x: x.mean() * 100).reset_index()
                .rename(columns={"Profitable": "Win Rate %"})
            )
            fig_wr = px.bar(
                wr_sig, x="Signal", y="Win Rate %",
                title="Win Rate % by Signal Type", color="Signal",
                color_discrete_map={"Bullish": "green", "Neutral": "gold", "Weak": "red"}
            )
            fig_wr.update_layout(height=350)
            st.plotly_chart(fig_wr, use_container_width=True)

            st.markdown("**Distribution of signals saved**")
            st.caption("How many of your saved snapshots were Bullish, Neutral, or Weak?")
            sig_dist = bt_df["Signal"].value_counts().reset_index()
            sig_dist.columns = ["Signal", "Count"]
            fig_dist = px.pie(
                sig_dist, names="Signal", values="Count",
                title="Signal Distribution", color="Signal",
                color_discrete_map={"Bullish": "green", "Neutral": "gold", "Weak": "red"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Top performing tickers**")
                st.caption("Highest hypothetical returns since snapshot.")
                top5 = valid.nlargest(5, "Return %")[["Ticker", "Signal", "Return %"]]
                fig_top = px.bar(
                    top5, x="Ticker", y="Return %",
                    color="Return %", color_continuous_scale="Greens",
                    title="Top 5 Hypothetical Returns"
                )
                fig_top.update_layout(height=300)
                st.plotly_chart(fig_top, use_container_width=True)
            with col_b:
                st.markdown("**Worst performing tickers**")
                st.caption("Lowest hypothetical returns since snapshot.")
                bot5 = valid.nsmallest(5, "Return %")[["Ticker", "Signal", "Return %"]]
                fig_bot = px.bar(
                    bot5, x="Ticker", y="Return %",
                    color="Return %", color_continuous_scale="Reds_r",
                    title="Bottom 5 Hypothetical Returns"
                )
                fig_bot.update_layout(height=300)
                st.plotly_chart(fig_bot, use_container_width=True)

            render_disclaimer(DISCLAIMER)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stock Discovery
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stock Discovery Scanner")
    st.info(
        "📡 This scanner uses **yfinance only** — no Alpha Vantage, Finnhub, or OpenAI calls. "
        "Scan freely without worrying about API limits. "
        "After finding a candidate, run Phase 2 analysis manually in the Deep Dive tab. "
        "For educational research only. Not financial advice."
    )

    category     = st.selectbox("Choose a category to scan", get_category_names())
    scan_tickers = get_category_tickers(category)
    st.caption(
        f"Scanning {len(scan_tickers)} tickers in **{category}**: {', '.join(scan_tickers)}"
    )

    if st.button(f"Scan {category}", key="run_scan"):
        prog            = st.progress(0)
        status_text     = st.empty()
        partial_results = []

        for i, ticker in enumerate(scan_tickers):
            status_text.caption(f"Scanning {ticker} ({i+1}/{len(scan_tickers)})...")
            prog.progress((i + 1) / len(scan_tickers))
            result = get_quick_stats(
                ticker, get_data, add_indicators,
                score_stock, get_signal_short, safe_float
            )
            if result:
                partial_results.append({
                    "Ticker":         result["ticker"],
                    "Close":          f"${result['close']:.2f}",
                    "RSI":            f"{result['rsi']:.1f}",
                    "MA20":           f"${safe_float(result['latest']['MA20']):.2f}",
                    "MA50":           f"${safe_float(result['latest']['MA50']):.2f}",
                    "Volume":         f"{safe_float(result['latest']['Volume']):,.0f}",
                    "Research Score": result["score"],
                    "Signal":         result["signal"],
                    "Top Signal":     result["reasons"][0] if result["reasons"] else "—",
                })

        prog.empty()
        status_text.empty()

        if partial_results:
            st.session_state["scan_results"]  = partial_results
            st.session_state["scan_category"] = category

    if "scan_results" in st.session_state:
        scan_rows  = st.session_state["scan_results"]
        scan_df    = pd.DataFrame(scan_rows).sort_values(
            "Research Score", ascending=False
        ).reset_index(drop=True)
        display_df = build_display_df(scan_rows)

        st.subheader(f"Scan Results — {st.session_state.get('scan_category', category)}")
        st.caption("Ranked by research score. Educational metric only. Not a trade recommendation.")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        render_disclaimer(DISCLAIMER)

        st.divider()
        st.subheader("Send a Candidate to Deep Dive")
        st.caption("Select a ticker to investigate further. Then go to Deep Dive + Phase 2 tab.")
        chosen = st.selectbox("Select a ticker", scan_df["Ticker"].tolist())
        col_a, col_b = st.columns(2)

        if col_a.button(f"Add {chosen} to watchlist"):
            current_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if chosen in current_list:
                st.info(f"{chosen} is already in your watchlist.")
            else:
                st.success(
                    f"**{chosen}** ready to add. Update the sidebar ticker input to include it, "
                    f"then select it in the Deep Dive dropdown."
                )

        if col_b.button(f"Quick stats for {chosen}"):
            stats = get_quick_stats(
                chosen, get_data, add_indicators,
                score_stock, get_signal_short, safe_float
            )
            if stats:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Close",          f"${stats['close']:.2f}")
                q2.metric("RSI",            f"{stats['rsi']:.1f}")
                q3.metric("Research Score", f"{stats['score']}/100")
                q4.metric("Signal",         stats["signal"])
                for r in stats["reasons"]:
                    st.write(f"• {r}")
                render_disclaimer(DISCLAIMER)
            else:
                st.warning(f"Could not load data for {chosen}")

        st.info(
            f"**Next steps for {chosen}:**\n"
            "1. Add to watchlist in the sidebar\n"
            "2. Go to Deep Dive + Phase 2 tab\n"
            "3. Fetch news sentiment (1 Alpha Vantage request)\n"
            "4. Fetch insider disclosures (1 Finnhub request)\n"
            "5. Check SEC filings in Public Disclosures tab\n"
            "6. Use free ChatGPT prompt or OpenAI button for AI analysis\n"
            "7. Log a paper trade if the setup looks interesting\n\n"
            "⚠️ All analysis is for educational research only. Not financial advice."
        )
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Market Sentiment Scanner
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Market Sentiment Scanner")
    st.info(
        "📡 This scanner fetches broad market news from Alpha Vantage. "
        "It fetches up to **3 pages of articles** (3 requests) covering the last 1–2 days "
        "of news, surfaces the **top 20 most-mentioned tickers**, and shows "
        "**which sectors are getting the most coverage**. "
        "For educational research only. Not financial advice."
    )

    av_rem  = requests_remaining("alpha_vantage", AV_DAILY_LIMIT)
    av_used = get_usage_today("alpha_vantage")

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Alpha Vantage Remaining", f"{av_rem} / {AV_DAILY_LIMIT}")
    sc2.metric("Used Today",              str(av_used))
    sc3.metric("This scan costs",         "up to 3 requests")
    st.progress(min(av_used / AV_DAILY_LIMIT, 1.0))

    if av_rem <= 0:
        st.error("No Alpha Vantage requests left today. Resets at midnight.")
    else:
        if av_rem < 3:
            st.error(
                f"Only {av_rem} request(s) left — scan will fetch as many pages as possible."
            )
        elif av_rem <= 5:
            st.warning(f"Only {av_rem} requests left today — scan uses up to 3.")
        elif av_rem <= 10:
            st.warning(f"{av_rem} Alpha Vantage requests left today.")

        # Let user choose how many pages
        pages_to_fetch = st.slider(
            "Pages to fetch (each page = 1 request, ~50 articles)",
            min_value=1, max_value=min(3, av_rem),
            value=min(3, av_rem),
            help="More pages = more tickers found, covers 1-2 days back. Each page costs 1 request."
        )
        st.caption(
            f"Fetching **{pages_to_fetch} page(s)** = ~{pages_to_fetch * 50} articles. "
            f"Covers approximately {'today only' if pages_to_fetch == 1 else '1–2 days'} of news. "
            "Results cached for 30 minutes — refreshing the page does not cost more requests."
        )

        if st.button(
            f"Confirm — Run Market Sentiment Scan ({pages_to_fetch} request(s))",
            key="run_market_scan"
        ):
            with st.spinner("Scanning market news across multiple article pages..."):
                scan_results, scan_err = cached_market_sentiment_scan(
                    alpha_key, pages_to_fetch
                )
            if scan_err:
                st.error(f"Scan failed: {scan_err}")
            else:
                st.session_state["market_scan_results"] = scan_results
                st.session_state["market_scan_time"]    = datetime.now().strftime("%Y-%m-%d %H:%M")

    if "market_scan_results" in st.session_state:
        results   = st.session_state["market_scan_results"]
        scan_time = st.session_state.get("market_scan_time", "unknown")

        pages_used    = results.get("pages_fetched", 1)
        total_arts    = results.get("total_articles", 0)
        ticker_list   = results.get("tickers", [])
        sector_list   = results.get("sector_summary", [])

        st.divider()
        st.caption(
            f"Scanned {total_arts} unique articles across {pages_used} page(s) — "
            f"last run: {scan_time}"
        )

        # ── Sector summary ────────────────────────────────────────────────────
        st.subheader("Sector Activity Summary")
        st.caption(
            "Which sectors are getting the most news coverage right now? "
            "Sorted by total article mentions. For research only."
        )

        if sector_list:
            sector_df = pd.DataFrame(sector_list)

            def sector_sentiment_color(val):
                if val == "Bullish": return "color: green"
                if val == "Bearish": return "color: red"
                return "color: gray"

            st.dataframe(
                sector_df.style.map(sector_sentiment_color, subset=["Sentiment"]),
                use_container_width=True, hide_index=True
            )

            # Sector mentions bar chart
            fig_sectors = px.bar(
                sector_df.head(10),
                x="Sector", y="Total Mentions",
                color="Sentiment",
                color_discrete_map={
                    "Bullish": "green", "Neutral": "gold", "Bearish": "red"
                },
                title="Top Sectors by Article Mentions",
                text="Total Mentions",
            )
            fig_sectors.update_layout(
                height=400,
                xaxis_tickangle=-30,
                showlegend=True,
            )
            fig_sectors.update_traces(textposition="outside")
            st.plotly_chart(fig_sectors, use_container_width=True)

        st.divider()

        # ── Top 20 tickers ────────────────────────────────────────────────────
        st.subheader(f"Top 20 Most-Covered Tickers — scanned {scan_time}")
        st.caption(
            "Ranked by number of article mentions then sentiment score. "
            "Includes tickers from up to 2 days of news. "
            "For research only — not a recommendation."
        )

        table_rows = []
        for r in ticker_list:
            t = r["Time"]
            if len(t) == 8:
                t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
            table_rows.append({
                "Ticker":        r["Ticker"],
                "Sector":        r["Sector"],
                "Mentions":      r["Mentions"],
                "Avg Sentiment": r["Avg Sentiment"],
                "Sentiment":     r["Sentiment"],
                "Top Headline":  (
                    r["Top Headline"][:75] + "..."
                    if len(r["Top Headline"]) > 75
                    else r["Top Headline"]
                ),
                "Source":        r["Source"],
                "Date":          t,
            })

        table_df = pd.DataFrame(table_rows)

        def sentiment_color(val):
            if val == "Bullish": return "color: green"
            if val == "Bearish": return "color: red"
            return "color: gray"

        st.dataframe(
            table_df.style.map(sentiment_color, subset=["Sentiment"]),
            use_container_width=True, hide_index=True
        )
        render_disclaimer(DISCLAIMER)

        # ── Filter by sector ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Filter by Sector")
        available_sectors = sorted(set(r["Sector"] for r in ticker_list))
        chosen_sector     = st.selectbox(
            "Show tickers from sector",
            ["All sectors"] + available_sectors,
            key="market_scan_sector_filter"
        )

        filtered_tickers = (
            ticker_list if chosen_sector == "All sectors"
            else [r for r in ticker_list if r["Sector"] == chosen_sector]
        )

        if filtered_tickers:
            st.caption(
                f"Showing {len(filtered_tickers)} ticker(s) "
                f"{'across all sectors' if chosen_sector == 'All sectors' else f'in {chosen_sector}'}."
            )
        else:
            st.info(f"No tickers found in {chosen_sector} from this scan.")

        # ── Explore a ticker ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Explore a Ticker from the Scan")
        st.caption("Select any ticker to see its news context and run a free technical check.")

        explore_options = [r["Ticker"] for r in filtered_tickers] if filtered_tickers else [r["Ticker"] for r in ticker_list]
        chosen_scan     = st.selectbox(
            "Select a ticker to explore",
            explore_options,
            key="market_scan_chosen"
        )

        chosen_data = next((r for r in ticker_list if r["Ticker"] == chosen_scan), None)
        if chosen_data:
            st.markdown(f"**News context for {chosen_scan}:**")
            nc1, nc2, nc3, nc4 = st.columns(4)
            nc1.metric("Mentions",      chosen_data["Mentions"])
            nc2.metric("Avg Sentiment", chosen_data["Avg Sentiment"])
            nc3.metric("Sentiment",     chosen_data["Sentiment"])
            nc4.metric("Sector",        chosen_data["Sector"])
            t = chosen_data["Time"]
            if len(t) == 8:
                t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
            st.markdown(
                f"**Top headline:** [{chosen_data['Top Headline']}]({chosen_data['URL']})  \n"
                f"*{chosen_data['Source']} · {t}*"
            )

        # ── Quick technical check ─────────────────────────────────────────────
        st.divider()
        st.markdown(f"**Quick technical check for {chosen_scan}** (yfinance only — free)")
        if st.button(f"Run technical scan for {chosen_scan}", key="market_scan_tech"):
            stats = get_quick_stats(
                chosen_scan, get_data, add_indicators,
                score_stock, get_signal_short, safe_float
            )
            if stats is None:
                st.error(f"No price data found for {chosen_scan}")
            else:
                qt1, qt2, qt3, qt4 = st.columns(4)
                qt1.metric("Close",          f"${stats['close']:.2f}")
                qt2.metric("RSI",            f"{stats['rsi']:.1f}")
                qt3.metric("Research Score", f"{stats['score']}/100")
                qt4.metric("Signal",         stats["signal"])
                render_signal_banner(stats["score"], get_signal(stats["score"]))

                fig_scan = go.Figure()
                fig_scan.add_trace(go.Candlestick(
                    x=stats["data"].index,
                    open=stats["data"]["Open"],
                    high=stats["data"]["High"],
                    low=stats["data"]["Low"],
                    close=stats["data"]["Close"],
                    name="Price"
                ))
                fig_scan.add_trace(go.Scatter(
                    x=stats["data"].index,
                    y=stats["data"]["MA20"],
                    name="MA20"
                ))
                fig_scan.add_trace(go.Scatter(
                    x=stats["data"].index,
                    y=stats["data"]["MA50"],
                    name="MA50"
                ))
                fig_scan.update_layout(
                    height=400,
                    xaxis_rangeslider_visible=False,
                    title=f"{chosen_scan} — 3 Month Chart (data may be delayed ~15 min)"
                )
                st.plotly_chart(fig_scan, use_container_width=True)
                render_score_breakdown(stats["reasons"], DISCLAIMER)

        # ── Send to Deep Dive ─────────────────────────────────────────────────
        st.divider()
        st.markdown(f"**Send {chosen_scan} to Deep Dive for full Phase 2 analysis**")
        st.caption(
            "Adding to your watchlist lets you fetch news sentiment, insider disclosures, "
            "SEC filings, and run the AI summary in the Deep Dive tab."
        )
        if st.button(f"Add {chosen_scan} to watchlist", key="market_scan_add"):
            current_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if chosen_scan in current_list:
                st.info(f"{chosen_scan} is already in your watchlist.")
            else:
                new_input = tickers_input.rstrip(", ") + f", {chosen_scan}"
                st.success(
                    f"**{chosen_scan}** is ready to add. "
                    f"Update your sidebar watchlist input to:  \n"
                    f"`{new_input}`  \n"
                    f"Then select **{chosen_scan}** in the Deep Dive dropdown."
                )

        st.info(
            f"**Full research flow for {chosen_scan}:**\n"
            "1. Add to sidebar watchlist\n"
            "2. Go to Deep Dive + Phase 2 tab\n"
            "3. Fetch news sentiment (1 Alpha Vantage request)\n"
            "4. Fetch insider disclosures (1 Finnhub request)\n"
            "5. Check SEC filings in Public Disclosures tab\n"
            "6. Use free ChatGPT prompt or OpenAI button for AI analysis\n"
            "7. Log a paper trade if the setup looks interesting\n\n"
            "⚠️ All analysis is for educational research only. Not financial advice."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Public Disclosures
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Public Disclosures — SEC EDGAR")
    st.info(
        "📋 **About this tab:** All data comes directly from the SEC's public EDGAR database. "
        "No non-public or confidential information is accessed. "
        "Form 4 insiders have up to 2 business days to file. "
        "13F institutional filings are delayed by 45 days. "
        "For educational research only. Not financial advice."
    )
    st.caption(
        "⚙️ SEC fair-access guidelines allow a maximum of 10 requests per second. "
        "This app uses manual confirmation gates and SQLite caching to stay well under that limit. "
        "SEC User-Agent is configured securely through Streamlit Secrets."
    )

    st.subheader("Step 1 — Company Lookup")
    st.caption(
        "Enter a ticker to find its SEC CIK number and company name. "
        "Results are cached in SQLite — repeat lookups are free and instant."
    )

    lookup_col1, lookup_col2 = st.columns([2, 1])
    lookup_ticker = lookup_col1.text_input(
        "Ticker to look up", value=selected, key="sec_lookup_ticker"
    ).upper().strip()

    sec_rem = requests_remaining("sec", SEC_DAILY_LIMIT)
    lookup_col2.metric("SEC Requests Left", f"{sec_rem} / {SEC_DAILY_LIMIT}")

    cached_info  = load_sec_company_cache(lookup_ticker)
    company_info = None

    if cached_info:
        st.success(
            f"Loaded from cache (fetched {cached_info['fetched_date']}) — no SEC request used."
        )
        company_info = cached_info
    else:
        st.caption(
            f"No cached data for **{lookup_ticker}**. "
            f"Looking up will use 1 SEC request ({sec_rem - 1} remaining after)."
        )
        if sec_rem <= 0:
            st.error("No SEC requests left today. Resets at midnight.")
        elif st.button(
            f"Confirm — look up {lookup_ticker} on SEC EDGAR",
            key="sec_lookup_btn"
        ):
            with st.spinner(f"Looking up {lookup_ticker} on SEC EDGAR..."):
                company_info, err = get_sec_company_info(
                    lookup_ticker, sec_agent, increment_usage,
                    save_sec_company_cache, load_sec_company_cache,
                )
            if err:
                st.error(f"Lookup failed: {err}")
                company_info = None
            else:
                st.rerun()

    if company_info:
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Company Name", company_info["company_name"])
        r2.metric("Ticker",       lookup_ticker)
        r3.metric("CIK",          company_info["cik"])

        sec_base = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={company_info['cik']}"
        )
        st.markdown(
            f"🔗 [View all SEC filings for {company_info['company_name']}]"
            f"({sec_base}&type=&dateb=&owner=include&count=40&search_text=)  \n"
            f"🔗 [SEC submissions JSON]({company_info['submissions_url']})"
        )
        st.caption("Links open the official SEC EDGAR website.")

        st.divider()
        st.subheader("Step 2 — Recent SEC Filings")
        st.caption("Results cached for today — repeat fetches are free.")

        with st.expander("What do these SEC form types mean?", expanded=False):
            st.markdown("""
| Form | What it means |
|---|---|
| **Form 4** | Insider transaction disclosure — filed within 2 business days of the transaction. |
| **8-K** | Major company event — earnings surprises, CEO changes, mergers. Filed within 4 business days. |
| **10-Q** | Quarterly financial report — unaudited, filed within 40–45 days after quarter end. |
| **10-K** | Annual financial report — audited, filed within 60–90 days after fiscal year end. |
| **13F-HR** | Institutional holdings — filed quarterly, delayed up to 45 days after quarter end. |
| **13F-HR/A** | Amended institutional holdings report. |
""")
            st.caption(
                "All forms are publicly required disclosures. "
                "Always check filed date vs report date. For educational research only."
            )

        form_filter = st.multiselect(
            "Filter by form type",
            options=["4", "10-K", "10-Q", "8-K", "13F-HR", "13F-HR/A"],
            default=["4", "10-K", "10-Q", "8-K"],
            key="sec_form_filter"
        )

        cached_filings = load_sec_filings_cache(lookup_ticker)
        filings        = None

        if cached_filings:
            filings = cached_filings
            st.success("Filings loaded from cache — no SEC request used.")
        else:
            st.caption(
                f"No cached filings for **{lookup_ticker}** today. "
                f"Fetching will use 1 SEC request ({sec_rem - 1} remaining after)."
            )
            if sec_rem <= 0:
                st.error("No SEC requests left today.")
            elif st.button(
                f"Confirm — fetch recent SEC filings for {lookup_ticker}",
                key="sec_filings_btn"
            ):
                with st.spinner(f"Fetching SEC filings for {lookup_ticker}..."):
                    filings, err, _ = get_sec_filings(
                        lookup_ticker, company_info["cik"],
                        sec_agent, increment_usage,
                        save_sec_filings_cache, load_sec_filings_cache,
                        form_types=None,
                    )
                if err:
                    st.error(f"Could not load filings: {err}")
                    filings = None
                else:
                    st.rerun()

        if filings:
            filtered = [f for f in filings if not form_filter or f["Form"] in form_filter]
            st.caption(
                f"Showing {len(filtered)} filings (filtered from {len(filings)} total). "
                "Click links to view official SEC documents."
            )
            if filtered:
                st.dataframe(
                    pd.DataFrame([{
                        "Form":        f["Form"],
                        "Filed":       f["Filed"],
                        "Report Date": f["Report Date"],
                        "Accession":   f["Accession"],
                    } for f in filtered]),
                    use_container_width=True, hide_index=True
                )
                st.markdown("**Filing links:**")
                for f in filtered[:15]:
                    col_f1, col_f2 = st.columns([3, 1])
                    col_f1.markdown(
                        f"**{f['Form']}** — Filed: {f['Filed']} | "
                        f"Report: {f['Report Date']}  \n"
                        f"Accession: `{f['Accession']}`"
                    )
                    col_f2.markdown(
                        f"[View Filing]({f['Filing URL']}) | [Index]({f['Index URL']})"
                        if f["Index URL"] else
                        f"[View Filing]({f['Filing URL']})"
                    )
                    st.divider()
                st.caption(
                    "All links open official SEC EDGAR pages. "
                    "Data is publicly available and may be delayed per SEC filing requirements."
                )
            else:
                st.info("No filings found for the selected form types.")

        render_disclaimer(DISCLAIMER)
        st.caption(
            "📋 All SEC data shown here is publicly available through SEC EDGAR. "
            "This tool accesses no non-public or confidential information. "
            "For educational research only."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Paper Trade Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Paper Trade Log")
    st.info(
        "📋 **Paper Trading Only:** All trades logged here are hypothetical and for educational "
        "purposes only. No real money is involved. Not financial advice."
    )

    with st.expander("Log a new paper trade"):
        st.caption("Track hypothetical trades to see how your research signals perform over time.")
        c1, c2, c3 = st.columns(3)
        trade_ticker = c1.text_input("Ticker", value=selected)
        trade_action = c2.selectbox("Action", ["BUY (paper)", "SELL (paper)"])
        trade_action_clean = trade_action.replace(" (paper)", "")

        raw_log = get_data(trade_ticker, "5d")
        default_price = float(raw_log["Close"].iloc[-1]) if not raw_log.empty else 100.0
        trade_price   = c3.number_input("Hypothetical Entry Price", value=default_price)

        raw_ind = get_data(trade_ticker, period)
        if not raw_ind.empty:
            ind_data        = add_indicators(raw_ind.copy())
            ind_latest      = ind_data.iloc[-1]
            trade_score, _  = score_stock(ind_data)
            trade_signal    = get_signal_short(trade_score)
            trade_rsi       = safe_float(ind_latest["RSI"])
            trade_ma20      = safe_float(ind_latest["MA20"])
            trade_ma50      = safe_float(ind_latest["MA50"])
        else:
            trade_score = 0;  trade_signal = "Unknown"
            trade_rsi   = 0.0; trade_ma20 = 0.0; trade_ma50 = 0.0

        trade_notes = st.text_input("Research notes (optional)")
        st.caption(
            f"Will save with: Score {trade_score}/100 | "
            f"RSI {trade_rsi:.1f} | Signal: {trade_signal}"
        )

        if st.button("Log Paper Trade"):
            save_trade(
                entry_date  = datetime.now().strftime("%Y-%m-%d %H:%M"),
                ticker      = trade_ticker,
                action      = trade_action_clean,
                entry_price = trade_price,
                ai_score    = trade_score,
                rsi         = trade_rsi,
                ma20        = trade_ma20,
                ma50        = trade_ma50,
                signal      = trade_signal,
                notes       = trade_notes,
            )
            st.success(
                f"Paper trade logged: {trade_action_clean} {trade_ticker} "
                f"at ${trade_price:.2f} (hypothetical)"
            )
            st.rerun()

    trades_df = load_trades()

    if trades_df.empty:
        st.info("No paper trades logged yet.")
    else:
        st.subheader("Open Paper Trades")
        st.caption("Gain/Loss is hypothetical. For educational tracking only.")
        open_trades = trades_df[trades_df["status"] == "Open"].copy()

        if not open_trades.empty:
            perf_rows = []
            for _, row in open_trades.iterrows():
                current_price = get_current_price(row["ticker"])
                if current_price and row["entry_price"] > 0:
                    gain_pct = (
                        (current_price - row["entry_price"]) / row["entry_price"] * 100
                    )
                    if row["action"] == "SELL":
                        gain_pct = -gain_pct
                    gain_str = f"{gain_pct:+.2f}% (hypothetical)"
                else:
                    current_price = 0.0
                    gain_str      = "N/A"
                try:
                    days_open = (
                        datetime.now() -
                        datetime.strptime(row["entry_date"], "%Y-%m-%d %H:%M")
                    ).days
                except:
                    days_open = 0
                perf_rows.append({
                    "ID":               row["id"],
                    "Date":             row["entry_date"],
                    "Ticker":           row["ticker"],
                    "Action":           row["action"],
                    "Entry $":          f"${row['entry_price']:.2f}",
                    "Current $":        f"${current_price:.2f}",
                    "Hypothetical P/L": gain_str,
                    "Days Open":        days_open,
                    "Score@Entry":      f"{row['ai_score']}/100",
                    "RSI@Entry":        f"{row['rsi']:.1f}",
                    "Signal@Entry":     row["signal"],
                    "Notes":            row["notes"],
                })

            perf_df = pd.DataFrame(perf_rows)
            st.dataframe(
                perf_df.drop(columns=["ID"]),
                use_container_width=True, hide_index=True
            )
            render_disclaimer(DISCLAIMER)

            st.subheader("Manage paper trades")
            trade_ids    = open_trades["id"].tolist()
            trade_labels = [
                f"{r['ticker']} {r['action']} @ ${r['entry_price']:.2f} ({r['entry_date']})"
                for _, r in open_trades.iterrows()
            ]
            sel_label = st.selectbox("Select a trade", trade_labels)
            sel_id    = trade_ids[trade_labels.index(sel_label)]
            mc1, mc2  = st.columns(2)
            if mc1.button("Mark as Closed"):
                close_trade(sel_id)
                st.success("Marked as closed.")
                st.rerun()
            if mc2.button("Delete Trade"):
                delete_trade(sel_id)
                st.success("Deleted.")
                st.rerun()

        closed_trades = trades_df[trades_df["status"] == "Closed"]
        if not closed_trades.empty:
            st.subheader("Closed Paper Trades")
            st.dataframe(
                closed_trades[[
                    "entry_date", "ticker", "action", "entry_price",
                    "ai_score", "rsi", "signal", "notes"
                ]].rename(columns={
                    "entry_date":  "Date",
                    "ticker":      "Ticker",
                    "action":      "Action",
                    "entry_price": "Entry $",
                    "ai_score":    "Score@Entry",
                    "rsi":         "RSI@Entry",
                    "signal":      "Signal@Entry",
                    "notes":       "Notes",
                }),
                use_container_width=True, hide_index=True
            )
            render_disclaimer(DISCLAIMER)

    st.divider()
    st.caption(
        "📋 All entries are hypothetical paper trades for educational research only. "
        "No real money is tracked here. Not financial advice."
    )
