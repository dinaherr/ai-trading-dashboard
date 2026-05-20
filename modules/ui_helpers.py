import streamlit as st

# ── Request gate ──────────────────────────────────────────────────────────────
def render_request_gate(api_name, limit, ticker, label, key_prefix,
                         get_usage_fn, requests_remaining_fn):
    used = get_usage_fn(api_name)
    rem  = requests_remaining_fn(api_name, limit)

    c1, c2, c3 = st.columns(3)
    c1.metric("Remaining Today", f"{rem} / {limit}")
    c2.metric("Used Today",      str(used))
    c3.metric("This call costs", "1 request")
    st.progress(min(used / limit, 1.0))

    if rem <= 0:
        st.error(f"No {label} requests left today. Resets at midnight.")
        return False
    if rem <= 5:
        st.error(f"Only {rem} {label} requests left!")
    elif rem <= 10:
        st.warning(f"{rem} {label} requests left today")

    st.info(
        f"Fetching **{label}** data for **{ticker}** costs 1 request "
        f"({rem - 1} remaining after)."
    )
    return st.button(
        f"Confirm — fetch {label} for {ticker}",
        key=f"{key_prefix}_confirm"
    )


# ── OpenAI gate ───────────────────────────────────────────────────────────────
def render_openai_gate(key_prefix, label, openai_daily_limit,
                        get_usage_fn, requests_remaining_fn):
    used = get_usage_fn("openai")
    rem  = requests_remaining_fn("openai", openai_daily_limit)

    st.caption(
        f"OpenAI: {used} used / {rem} remaining today "
        f"(self-set limit: {openai_daily_limit} — resets midnight)"
    )

    if rem <= 0:
        st.error("You've hit your self-set OpenAI limit for today.")
        return False
    if rem <= 3:
        st.error(f"Only {rem} OpenAI calls left today!")

    return st.button(
        f"Use OpenAI — {label} (1 call + tokens)",
        key=key_prefix
    )


# ── Signal banner ─────────────────────────────────────────────────────────────
def render_signal_banner(score, signal_full):
    if score >= 70:
        st.success(f"Signal alignment: {signal_full}")
    elif score >= 50:
        st.info(f"Signal alignment: {signal_full}")
    else:
        st.warning(f"Signal alignment: {signal_full}")


# ── Sentiment banner ──────────────────────────────────────────────────────────
def render_sentiment_banner(sentiment_label):
    if sentiment_label == "Bullish":
        st.success("News sentiment leaning bullish — signals may be favorable")
    elif sentiment_label == "Bearish":
        st.error("News sentiment leaning bearish — signals may be unfavorable")
    else:
        st.info("News sentiment neutral — no strong directional signal")


# ── Insider banner ────────────────────────────────────────────────────────────
def render_insider_banner(insider_signal, net_shares):
    if insider_signal == "Bullish":
        st.success(
            f"Open-market insider buying — net {net_shares:,} shares. "
            "Historically considered a potentially positive signal."
        )
    elif insider_signal == "Bearish":
        st.warning(
            f"Net insider selling — {net_shares:,} shares. "
            "Note: selling is often routine, planned, or for personal liquidity."
        )
    else:
        st.info(f"Mixed or neutral insider activity. Net shares: {net_shares:,}")


# ── API budget sidebar ────────────────────────────────────────────────────────
def render_api_budget_sidebar(api_configs, get_usage_fn, requests_remaining_fn):
    st.header("API Budget Today")
    for api_name, label, limit in api_configs:
        used = get_usage_fn(api_name)
        rem  = requests_remaining_fn(api_name, limit)
        st.metric(label, f"{rem} / {limit} left")
        st.progress(min(used / limit, 1.0))
        if rem <= 5:
            st.error(f"Only {rem} {label} requests left!")
        elif rem <= 10:
            st.warning(f"{rem} {label} requests left today")
    st.caption("All counters reset at midnight.")


# ── Score breakdown expander ──────────────────────────────────────────────────
def render_score_breakdown(reasons, disclaimer):
    with st.expander("Score breakdown (research signals only)"):
        st.caption("Educational signal indicators only. Do not predict future performance.")
        for r in reasons:
            st.write(f"• {r}")
        st.caption(disclaimer)


# ── Snapshot status ───────────────────────────────────────────────────────────
def render_snapshot_status(snap_result, ticker):
    if snap_result == "saved":
        st.success(f"📸 Today's signal snapshot saved for {ticker}.")
    else:
        st.caption(f"📸 Snapshot for {ticker} already exists for today.")


# ── Disclaimer caption ────────────────────────────────────────────────────────
def render_disclaimer(disclaimer):
    st.caption(disclaimer)
