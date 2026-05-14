# ============================================================
# app.py — Main entry point
# AI-Driven Stock Market Intelligence Suite
#
# Run with:  streamlit run app.py
# ============================================================

import os
import io
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# ── Internal modules ──────────────────────────────────────────
from data_engine import (fetch_stock_data, fetch_stock_info, enrich_data,
                          screen_bearish, run_deep_screen,
                          add_volume_profile, add_atr, compute_risk_reward,
                          fetch_market_regime, fetch_sector_performance,
                          compute_weighted_composite)
from ml_engine   import train_and_predict
from news_service import fetch_news
from ai_analyst   import ask_ai_analyst

# ── Environment ───────────────────────────────────────────────
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StockIQ — AI Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════
#  CUSTOM CSS — Glassmorphism Dark Theme
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #111827;
    --bg-card:       rgba(255,255,255,0.04);
    --border-color:  rgba(255,255,255,0.08);
    --accent-green:  #00d4aa;
    --accent-red:    #ff4d6d;
    --accent-blue:   #4f9cf9;
    --accent-amber:  #fbbf24;
    --text-primary:  #f0f4ff;
    --text-secondary:#8892a4;
    --glass-blur:    blur(16px);
    --radius:        10px;
    --shadow:        0 8px 32px rgba(0,0,0,0.4);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main container ── */
.main .block-container {
    col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("From", value=pd.to_datetime("2025-01-01"))
with col2:
    end_date = st.date_input("To", value=pd.to_datetime("today")) {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ── Glass card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.glass-card:hover { border-color: rgba(79,156,249,0.3); }

/* ── Metric cards override ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"]  { color: var(--text-secondary) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"]  { color: var(--text-primary)   !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    background: rgba(79,156,249,0.08) !important;
}

/* ── Sidebar inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"]  input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stDateInput"]  input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(79,156,249,0.15) !important;
}

/* ── Buttons ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #1e3a5f, #2d5a8e) !important;
    color: #fff !important;
    border: 1px solid rgba(79,156,249,0.4) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #2d5a8e, #3d7ac0) !important;
    box-shadow: 0 4px 16px rgba(79,156,249,0.3) !important;
}

/* ── Sentiment badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-pos { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.badge-neg { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.badge-neu { background: rgba(255,255,255,0.08); color: #8892a4; border: 1px solid rgba(255,255,255,0.12); }

/* ── Prediction card ── */
.pred-card {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(79,156,249,0.08));
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,212,170,0.1);
}
.pred-price  { font-size: 2.4rem; font-weight: 700; color: #00d4aa; }
.pred-label  { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.4rem; }
.pred-mae    { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; }

/* ── News card ── */
.news-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: rgba(79,156,249,0.3); }
.news-title  { font-size: 0.9rem; font-weight: 500; color: var(--text-primary); margin-bottom: 0.35rem; }
.news-meta   { font-size: 0.72rem; color: var(--text-secondary); }

/* ── Section header ── */
.sec-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0.8rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-color);
}

/* ── Bearish screener card ── */
.bear-card {
    background: rgba(255,77,109,0.05);
    border: 1px solid rgba(255,77,109,0.2);
    border-radius: var(--radius);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.bear-card:hover { border-color: rgba(255,77,109,0.4); }
.bear-ticker { font-size: 1rem; font-weight: 700; color: #ff4d6d; }
.bear-meta   { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.2rem; }
.bear-reason { font-size: 0.75rem; color: #fbbf24; margin-top: 0.3rem; }

/* ── Bullish screener card ── */
.bull-card {
    background: rgba(0,212,170,0.05);
    border: 1px solid rgba(0,212,170,0.22);
    border-radius: var(--radius);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.bull-card:hover { border-color: rgba(0,212,170,0.4); }
.bull-ticker { font-size: 1rem; font-weight: 700; color: #00d4aa; }
.bull-meta   { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.2rem; }
.bull-reason { font-size: 0.75rem; color: #4f9cf9; margin-top: 0.3rem; }

/* ── Time-horizon pill ── */
.horizon-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-right: 4px;
}
.pill-day  { background: rgba(79,156,249,0.15); color: #4f9cf9; border: 1px solid rgba(79,156,249,0.3); }
.pill-week { background: rgba(0,212,170,0.12);  color: #00d4aa; border: 1px solid rgba(0,212,170,0.25); }
.pill-next { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0 1.2rem;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f9cf9, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle { font-size: 0.78rem; color: var(--text-secondary); }

/* ── Error / info box ── */
.info-box {
    background: rgba(79,156,249,0.08);
    border: 1px solid rgba(79,156,249,0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.83rem;
    color: var(--text-secondary);
}

/* ── AI Chat styles ── */
.chat-msg-user {
    background: rgba(79,156,249,0.12);
    border: 1px solid rgba(79,156,249,0.25);
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 15%;
    font-size: 0.88rem;
    color: var(--text-primary);
}
.chat-msg-user::before {
    content: "🧑 You";
    display: block;
    font-size: 0.65rem;
    color: #4f9cf9;
    font-weight: 700;
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
}
.chat-msg-ai {
    background: rgba(0,212,170,0.06);
    border: 1px solid rgba(0,212,170,0.18);
    border-radius: 12px 12px 12px 2px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 15% 0.5rem 0;
    font-size: 0.88rem;
    color: var(--text-primary);
    line-height: 1.6;
}
.chat-msg-ai::before {
    content: "🤖 StockIQ AI";
    display: block;
    font-size: 0.65rem;
    color: #00d4aa;
    font-weight: 700;
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .main .block-container { padding: 0.8rem 0.8rem 2rem !important; }
    .pred-price  { font-size: 1.8rem; }
    .app-title   { font-size: 1.2rem; }
    .chat-msg-user { margin-left: 5%; }
    .chat-msg-ai   { margin-right: 5%; }
}

/* ── Risk Meter ── */
.risk-meter-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.risk-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin: 0.5rem 0 0.3rem;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}

/* ── Conviction Summary ── */
.conviction-box {
    background: linear-gradient(135deg,rgba(79,156,249,0.07),rgba(0,212,170,0.07));
    border: 1px solid rgba(79,156,249,0.22);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
}
.conviction-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #4f9cf9;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.conviction-score {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4aa;
    line-height: 1.1;
}
.conviction-label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem; }
.conviction-reasons { font-size: 0.75rem; color: #8892a4; margin-top: 0.45rem; line-height: 1.6; }

/* ── Volume quality badge ── */
.vol-confirmed  { background:rgba(0,212,170,0.15);  color:#00d4aa; border:1px solid rgba(0,212,170,0.3); }
.vol-low-rally  { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }
.vol-hi-decline { background:rgba(255,77,109,0.15); color:#ff4d6d; border:1px solid rgba(255,77,109,0.3); }
.vol-weak       { background:rgba(255,255,255,0.06); color:#8892a4; border:1px solid rgba(255,255,255,0.1); }

/* ── RR card ── */
.rr-card {
    background: rgba(79,156,249,0.06);
    border: 1px solid rgba(79,156,249,0.2);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
}
.rr-label { font-size:0.72rem; color:var(--text-secondary); }
.rr-value { font-size:1.1rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1.2rem'>
        <div style='font-size:2rem'>📈</div>
        <div style='font-size:1rem; font-weight:700; color:#4f9cf9;'>StockIQ</div>
        <div style='font-size:0.7rem; color:#8892a4;'>AI Market Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Search")
    ticker = st.text_input(
        "NSE Ticker Symbol",
        value="RELIANCE.NS",
        placeholder="e.g. RELIANCE.NS",
        help="Append .NS for NSE, .BO for BSE, or leave bare for US stocks.",
    ).upper().strip()

    st.markdown("### 📅 Date Range")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=365))
    with col_d2:
        end_date = st.date_input("To", value=date.today())

    st.markdown("### ⚙️ Indicators")
    show_ema  = st.toggle("EMA (20 & 50)",  value=True)
    show_rsi  = st.toggle("RSI (14)",        value=True)
    show_macd = st.toggle("MACD (12/26/9)",  value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#8892a4; line-height:1.6;'>
    📦 <b>Data:</b> yfinance &nbsp;|&nbsp; 🤖 <b>ML:</b> scikit-learn<br>
    📰 <b>News:</b> NewsAPI &nbsp;|&nbsp; 📊 <b>Charts:</b> Plotly
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, start, end, use_ema, use_rsi, use_macd):
    df   = fetch_stock_data(ticker, str(start), str(end))
    info = fetch_stock_info(ticker)
    if not df.empty:
        df = enrich_data(df, show_ema=use_ema, show_rsi=use_rsi, show_macd=use_macd)
    return df, info


@st.cache_data(ttl=600, show_spinner=False)
def load_ml(ticker):
    """Train ML on 2-year window regardless of sidebar date range."""
    two_yr_start = str(date.today() - timedelta(days=730))
    two_yr_end   = str(date.today())
    df_ml = fetch_stock_data(ticker, two_yr_start, two_yr_end)
    if df_ml.empty:
        return {"error": "Not enough data for ML training."}
    df_ml = enrich_data(df_ml, show_ema=True, show_rsi=True, show_macd=True)
    return train_and_predict(df_ml)


@st.cache_data(ttl=900, show_spinner=False)
def load_screener(lookback: int):
    """Run bearish screener (cached 15 min)."""
    return screen_bearish(lookback_days=lookback)


@st.cache_data(ttl=900, show_spinner=False)
def load_deep_screen():
    """Run full 10-year deep scan for both bullish & bearish (cached 15 min)."""
    return run_deep_screen(news_api_key=NEWS_API_KEY)


with st.spinner("🔍 Fetching market data..."):
    df, stock_info = load_data(ticker, start_date, end_date, show_ema, show_rsi, show_macd)

if df is None or (isinstance(df, pd.DataFrame) and df.empty):
    st.error(
        f"⚠️ Could not load data for '**{ticker}**'. Common fixes:\n"
        "- Indian NSE stocks: use .NS suffix (e.g. RELIANCE.NS)\n"
        "- BSE stocks: use .BO suffix (e.g. RELIANCE.BO)\n"
        "- Verify ticker at finance.yahoo.com\n"
        "- Try widening the date range\n"
        "- Wait a moment and retry (Yahoo Finance rate limit)"
    )
    st.stop()


# ════════════════════════════════════════════════════════════
#  APP HEADER
# ════════════════════════════════════════════════════════════
company_name = stock_info.get("longName", ticker)
sector       = stock_info.get("sector",   "N/A")
currency     = stock_info.get("currency", "₹")

st.markdown(f"""
<div class='app-header'>
    <div>
        <div class='app-title'>📈 {company_name}</div>
        <div class='app-subtitle'>
            <span style='color:#4f9cf9;'>{ticker}</span>
            &nbsp;·&nbsp;{sector}
            &nbsp;·&nbsp;Data from {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  KPI METRICS ROW
# ════════════════════════════════════════════════════════════
latest  = df.iloc[-1]
prev    = df.iloc[-2]
cur_px  = float(latest["Close"])
prev_px = float(prev["Close"])
chg_pct = ((cur_px - prev_px) / prev_px) * 100
volume  = int(latest["Volume"])
high52  = float(df["High"].max())
low52   = float(df["Low"].min())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price",  f"{currency} {cur_px:,.2f}")
c2.metric("Day's Change",   f"{chg_pct:+.2f}%",        delta=f"{cur_px - prev_px:+.2f}")
c3.metric("Volume",         f"{volume:,}")
c4.metric("52W High",       f"{currency} {high52:,.2f}")
c5.metric("52W Low",        f"{currency} {low52:,.2f}")


# ════════════════════════════════════════════════════════════
#  DOWNLOAD BUTTON
# ════════════════════════════════════════════════════════════
csv_buffer = io.StringIO()
df.to_csv(csv_buffer)
st.download_button(
    label="⬇️ Download Historical Data (CSV)",
    data=csv_buffer.getvalue(),
    file_name=f"{ticker}_{start_date}_{end_date}.csv",
    mime="text/csv",
)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab_dash, tab_tech, tab_ai, tab_news, tab_bear, tab_bull, tab_risk, tab_chat = st.tabs([
    "📊 Dashboard",
    "📉 Technical Analysis",
    "🤖 AI Prediction",
    "📰 News & Sentiment",
    "🔻 Bearish Screener",
    "🚀 Bullish Screener",
    "⚖️ Risk & Volume",
    "💬 AI Analyst Chat",
])


# ─── helpers ────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#f0f4ff", size=12),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
    margin=dict(l=0, r=0, t=40, b=0),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(autosize=True)
    return fig


# ════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("<div class='sec-header'>Candlestick Chart</div>", unsafe_allow_html=True)

    fig_main = go.Figure()

    # Candlestick
    fig_main.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff4d6d",
        increasing_fillcolor="rgba(0,212,170,0.7)",
        decreasing_fillcolor="rgba(255,77,109,0.7)",
    ))

    # EMA overlays
    if show_ema and "EMA_20" in df.columns:
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df["EMA_20"],
            name="EMA 20", line=dict(color="#4f9cf9", width=1.5),
        ))
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df["EMA_50"],
            name="EMA 50", line=dict(color="#fbbf24", width=1.5),
        ))

    fig_main.update_layout(
        title=f"{ticker} — Price Action",
        xaxis_rangeslider_visible=False,
        height=480,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Volume bar chart
    st.markdown("<div class='sec-header'>Trading Volume</div>", unsafe_allow_html=True)
    colors = ["#00d4aa" if c >= o else "#ff4d6d"
              for c, o in zip(df["Close"], df["Open"])]
    fig_vol = go.Figure(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="Volume", opacity=0.75,
    ))
    fig_vol.update_layout(height=200, title="Volume", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_vol, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — TECHNICAL ANALYSIS
# ════════════════════════════════════════════════════════════
with tab_tech:

    # ── RSI ──────────────────────────────────────────────────
    if show_rsi and "RSI" in df.columns:
        st.markdown("<div class='sec-header'>RSI — Relative Strength Index (14)</div>",
                    unsafe_allow_html=True)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name="RSI", line=dict(color="#4f9cf9", width=2),
            fill="tozeroy", fillcolor="rgba(79,156,249,0.07)",
        ))
        fig_rsi.add_hline(y=70, line=dict(color="#ff4d6d", width=1.2, dash="dash"),
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line=dict(color="#00d4aa", width=1.2, dash="dash"),
                          annotation_text="Oversold (30)")
        rsi_layout = {**PLOTLY_LAYOUT, "yaxis": dict(range=[0, 100],
                      showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)}
        fig_rsi.update_layout(height=280, title="RSI", **rsi_layout)
        st.plotly_chart(fig_rsi, use_container_width=True)

        latest_rsi = df["RSI"].iloc[-1]
        if latest_rsi > 70:
            rsi_msg = f"⚠️ RSI at **{latest_rsi:.1f}** — market appears **overbought**."
        elif latest_rsi < 30:
            rsi_msg = f"💡 RSI at **{latest_rsi:.1f}** — market appears **oversold**."
        else:
            rsi_msg = f"✅ RSI at **{latest_rsi:.1f}** — in **neutral** territory."
        st.markdown(f"<div class='info-box'>{rsi_msg}</div>", unsafe_allow_html=True)
    else:
        st.info("Enable RSI toggle in the sidebar to see this chart.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MACD ─────────────────────────────────────────────────
    if show_macd and "MACD" in df.columns:
        st.markdown("<div class='sec-header'>MACD (12 / 26 / 9)</div>",
                    unsafe_allow_html=True)

        hist_colors = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in df["MACD_Hist"]]

        fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                                 shared_xaxes=True, vertical_spacing=0.04)
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#4f9cf9", width=2)), row=1, col=1)
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#fbbf24", width=1.5)), row=1, col=1)
        fig_macd.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Histogram",
            marker_color=hist_colors, opacity=0.8), row=2, col=1)

        fig_macd.update_layout(height=380, title="MACD", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("Enable MACD toggle in the sidebar to see this chart.")

    # ── EMA Table ────────────────────────────────────────────
    if show_ema and "EMA_20" in df.columns:
        st.markdown("<div class='sec-header'>EMA Summary</div>", unsafe_allow_html=True)
        ema20 = round(df["EMA_20"].iloc[-1], 2)
        ema50 = round(df["EMA_50"].iloc[-1], 2)
        trend = "📈 Bullish (Price > EMA20 > EMA50)" if cur_px > ema20 > ema50 \
           else "📉 Bearish (Price < EMA20 < EMA50)" if cur_px < ema20 < ema50 \
           else "➡️ Mixed / Sideways"
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("EMA 20",  f"{currency} {ema20:,.2f}")
        ec2.metric("EMA 50",  f"{currency} {ema50:,.2f}")
        ec3.metric("Trend",   trend)


# ════════════════════════════════════════════════════════════
#  TAB 3 — AI PREDICTION
# ════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown("<div class='sec-header'>🤖 Next-Day Price Prediction (Random Forest)</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    The model trains on <b>2 years</b> of historical data using an 80/20 chronological
    train-test split. Features include lagged prices, rolling statistics, and all
    technical indicators. MAE is reported on the unseen 20% hold-out set.
    </div><br>
    """, unsafe_allow_html=True)

    with st.spinner("🧠 Training Random Forest model…"):
        ml_result = load_ml(ticker)

    if "error" in ml_result:
        st.error(f"ML Engine: {ml_result['error']}")
    else:
        pred   = ml_result["predicted_price"]
        mae    = ml_result["mae"]
        direct = ml_result["direction"]
        dir_emoji = "📈" if direct == "Up" else ("📉" if direct == "Down" else "➡️")
        dir_color = "#00d4aa" if direct == "Up" else ("#ff4d6d" if direct == "Down" else "#8892a4")

        # Prediction card
        st.markdown(f"""
        <div class='pred-card'>
            <div class='pred-label'>PREDICTED NEXT-DAY CLOSE</div>
            <div class='pred-price'>{currency} {pred:,.2f}</div>
            <div style='font-size:1rem; color:{dir_color}; margin:0.4rem 0;'>
                {dir_emoji} Expected direction: <b>{direct}</b>
            </div>
            <div class='pred-mae'>
                ⚖️ Model MAE (hold-out): ± {currency} {mae:,.2f}
                &nbsp;|&nbsp; Last Close: {currency} {ml_result['last_close']:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature Importance bar chart
        st.markdown("<div class='sec-header'>Feature Importance</div>", unsafe_allow_html=True)
        fi = ml_result["feature_importance"].head(10)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation="h",
            marker=dict(
                color=fi.values,
                colorscale=[[0, "#1e3a5f"], [1, "#00d4aa"]],
                showscale=False,
            ),
        ))
        fi_layout = {**PLOTLY_LAYOUT, "yaxis": dict(
            autorange="reversed",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
        )}
        fig_fi.update_layout(
            height=320,
            title="Top 10 Features by Importance",
            **fi_layout,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Disclaimer
        st.markdown("""
        <div class='info-box' style='font-size:0.75rem;'>
        ⚠️ <b>Disclaimer:</b> This prediction is generated by a machine learning model
        for educational purposes only. It is <b>not</b> financial advice.
        Past performance does not guarantee future results.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 — NEWS & SENTIMENT
# ════════════════════════════════════════════════════════════
with tab_news:
    st.markdown("<div class='sec-header'>📰 Latest News & Sentiment Analysis</div>",
                unsafe_allow_html=True)

    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        st.markdown("""
        <div class='info-box'>
        🔑 <b>NewsAPI key not configured.</b><br>
        Add your key to a <code>.env</code> file:<br>
        <code>NEWS_API_KEY=your_key_here</code><br><br>
        Get a free key at <a href='https://newsapi.org' target='_blank'
        style='color:#4f9cf9;'>newsapi.org</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("📡 Fetching news…"):
            articles = fetch_news(ticker, NEWS_API_KEY, n=5)

        if articles and "error" in articles[0]:
            st.warning(articles[0]["error"])
        else:
            # Sentiment summary
            labels = [a["sentiment"]["label"] for a in articles]
            pos_n  = labels.count("Positive")
            neg_n  = labels.count("Negative")
            neu_n  = labels.count("Neutral")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🟢 Positive", pos_n)
            sc2.metric("🔴 Negative", neg_n)
            sc3.metric("⚪ Neutral",  neu_n)
            st.markdown("<br>", unsafe_allow_html=True)

            for art in articles:
                sent  = art["sentiment"]
                badge_cls = {"Positive": "badge-pos",
                             "Negative": "badge-neg",
                             "Neutral":  "badge-neu"}[sent["label"]]

                pub = art["published_at"][:10] if art.get("published_at") else ""
                st.markdown(f"""
                <div class='news-card'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem;'>
                        <div class='news-title'><a href='{art["url"]}' target='_blank'
                            style='color:#f0f4ff; text-decoration:none;'>{art["title"]}</a>
                        </div>
                        <span class='badge {badge_cls}'>{sent["label"]}</span>
                    </div>
                    <div class='news-meta'>
                        📰 {art["source"]} &nbsp;·&nbsp; 📅 {pub}
                        &nbsp;·&nbsp; +{sent["pos"]} / -{sent["neg"]} words
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 5 — BEARISH SCREENER  (upgraded: 10yr + ML + News + horizons)
# ════════════════════════════════════════════════════════════
with tab_bear:
    st.markdown("<div class='sec-header'>🔻 Stocks Likely to Decline — Deep Bearish Screener</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This screener scans <b>25 popular NSE stocks</b> using <b>10 years of historical data</b>,
    a full suite of technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR, OBV),
    a <b>Random Forest ML signal</b>, and <b>live news sentiment</b>.
    It predicts which 5 stocks are most likely to fall <b>today, this week, and next week</b>.
    Only stocks with 2+ bearish signals are shown.<br><br>
    ⚠️ <b>Not financial advice.</b> Use as a starting point for further research only.
    </div>
    <br>
    """, unsafe_allow_html=True)

    # Controls row
    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        lookback = st.selectbox(
            "Lookback Window",
            options=[1, 2],
            index=1,
            format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
        )
    with bc2:
        run_screen = st.button("🔍 Run Bearish Screener", type="primary")

    if run_screen or "deep_results" not in st.session_state:
        with st.spinner("🔍 Deep-scanning 25 NSE stocks (10yr data · ML · News)… ~45s"):
            bull_r, bear_r = load_deep_screen()
            st.session_state["deep_bull_results"] = bull_r
            st.session_state["deep_bear_results"] = bear_r
            # legacy key for backward compatibility
            st.session_state["bear_results"]  = [
                {
                    "ticker":         r["ticker"],
                    "last_close":     r["last_close"],
                    "change_pct":     r["ret_1d"] if lookback == 1 else r["ret_5d"],
                    "rsi":            r["rsi"],
                    "signal_reasons": r["signal_reasons"],
                    "risk_level":     r["risk_level"],
                    "score":          r["score"],
                }
                for r in bear_r
            ]
            st.session_state["bear_lookback"] = lookback
            st.session_state["deep_results"]  = True

    results    = st.session_state.get("deep_bear_results", [])
    used_lb    = st.session_state.get("bear_lookback", lookback)

    if not results:
        st.success("✅ No strong bearish signals found across the watchlist right now.")
    else:
        top5 = results[:5]
        st.markdown(f"### 📉 Top {min(5, len(top5))} Bearish Stocks — Likely to Fall")

        # ── Time-horizon prediction summary ──────────────────
        st.markdown("""
        <div style='background:rgba(255,77,109,0.06); border:1px solid rgba(255,77,109,0.18);
             border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem;'>
            <div style='font-size:0.82rem; color:#ff4d6d; font-weight:700; margin-bottom:0.6rem;'>
                🔻 PREDICTED DECLINE HORIZONS
            </div>
            <div style='display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.8rem; color:#8892a4;'>
                <span><span class='horizon-pill pill-day'>TODAY</span> Based on 1-day momentum + ML signal</span>
                <span><span class='horizon-pill pill-week'>THIS WEEK</span> 5-day return + MACD + RSI pattern</span>
                <span><span class='horizon-pill pill-next'>NEXT WEEK</span> 20-day trend + 10yr history + news</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary bar chart (top 5)
        tickers_bear = [r["ticker"].replace(".NS", "") for r in top5]
        changes_week = [r.get("ret_5d", 0) for r in top5]
        bar_colors   = ["#ff4d6d" if c < 0 else "#fbbf24" for c in changes_week]

        fig_bear = go.Figure(go.Bar(
            x=tickers_bear,
            y=changes_week,
            marker_color=bar_colors,
            text=[f"{c:+.2f}%" for c in changes_week],
            textposition="outside",
            name="5D % Change",
        ))
        bear_layout = {**PLOTLY_LAYOUT, "yaxis": dict(
            title="5-Day % Change",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
        )}
        fig_bear.update_layout(
            title="5-Day Price Change — Top 5 Bearish Picks",
            height=300,
            **bear_layout,
        )
        st.plotly_chart(fig_bear, use_container_width=True)

        # Composite score comparison
        composites = [r.get("composite", 0) for r in top5]
        fig_comp = go.Figure(go.Bar(
            x=tickers_bear,
            y=composites,
            marker_color=["#ff4d6d" if c < 0 else "#fbbf24" for c in composites],
            text=[f"{c:+.0f}" for c in composites],
            textposition="outside",
            name="Composite Score",
        ))
        fig_comp.update_layout(
            title="Composite Score (−100 = most bearish, based on 10yr data + ML + news)",
            height=260,
            **{**PLOTLY_LAYOUT, "yaxis": dict(
                title="Score", range=[-110, 110],
                showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                zerolinecolor="rgba(255,255,255,0.15)",
            )},
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("<div class='sec-header'>Detailed Bearish Signals — Top 5</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed cards (top 5 only)
        for rank, r in enumerate(top5, 1):
            ret1d  = r.get("ret_1d", 0)
            ret5d  = r.get("ret_5d", 0)
            ret20d = r.get("ret_20d", 0)
            ml_sig = r.get("ml_pct", 0)
            news_s = r.get("news_score", 0)
            avg_yr = r.get("avg_yr_ret", 0)
            worst  = r.get("worst_yr", 0)
            pats   = ", ".join(r.get("patterns", [])) or "None detected"
            rsi_s  = f"RSI: {r['rsi']:.0f}" if r.get("rsi") is not None else ""
            reasons = " &nbsp;·&nbsp; ".join(r["signal_reasons"])

            day_color  = "#ff4d6d" if ret1d  < 0 else "#fbbf24"
            week_color = "#ff4d6d" if ret5d  < 0 else "#fbbf24"
            mo_color   = "#ff4d6d" if ret20d < 0 else "#fbbf24"
            ml_color   = "#ff4d6d" if ml_sig < 0 else "#00d4aa"
            news_color = "#ff4d6d" if news_s < 0 else ("#00d4aa" if news_s > 0 else "#8892a4")

            st.markdown(f"""
            <div class='bear-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-size:0.75rem; color:#8892a4; margin-right:6px;'>#{rank}</span>
                        <span class='bear-ticker'>{r['ticker']}</span>
                    </div>
                    <div style='display:flex; gap:6px; align-items:center;'>
                        <span style='font-size:0.75rem;'>{r.get('risk_level','')}</span>
                        <span style='font-size:0.75rem; color:#8892a4;'>Score: {r['score']}</span>
                    </div>
                </div>
                <div class='bear-meta' style='margin-top:0.5rem;'>
                    Last Close: <b>₹{r['last_close']:,.2f}</b>
                    &nbsp;·&nbsp; {rsi_s}
                    &nbsp;·&nbsp; Composite: <b>{r.get('composite', 0):+.0f}</b>
                    &nbsp;·&nbsp; Volatility: {r.get('volatility', 0):.1f}%
                </div>
                <div style='display:flex; gap:1.5rem; margin-top:0.55rem; flex-wrap:wrap; font-size:0.75rem;'>
                    <span>
                        <span class='horizon-pill pill-day'>TODAY</span>
                        <span style='color:{day_color}; font-weight:600;'>{ret1d:+.2f}%</span>
                        &nbsp; ML: <span style='color:{ml_color};'>{ml_sig:+.1f}%</span>
                    </span>
                    <span>
                        <span class='horizon-pill pill-week'>THIS WEEK</span>
                        <span style='color:{week_color}; font-weight:600;'>{ret5d:+.2f}%</span>
                    </span>
                    <span>
                        <span class='horizon-pill pill-next'>NEXT WEEK</span>
                        <span style='color:{mo_color}; font-weight:600;'>{ret20d:+.2f}%</span>
                        &nbsp; News: <span style='color:{news_color};'>{news_s:+.1f}</span>
                    </span>
                </div>
                <div style='font-size:0.72rem; color:#8892a4; margin-top:0.4rem;'>
                    📊 10Y Avg Annual: <b>{avg_yr:+.1f}%</b>
                    &nbsp;·&nbsp; Worst Year: <b style='color:#ff4d6d;'>{worst:.0f}%</b>
                    &nbsp;·&nbsp; Patterns: {pats}
                </div>
                <div class='bear-reason'>⚠️ {reasons}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(results) > 5:
            with st.expander(f"Show remaining {len(results) - 5} bearish stocks"):
                for r in results[5:]:
                    rsi_str  = f"RSI: {r['rsi']:.0f}" if r.get("rsi") is not None else ""
                    reasons  = " &nbsp;·&nbsp; ".join(r["signal_reasons"])
                    chg      = r.get("ret_5d", 0)
                    chg_color = "#ff4d6d" if chg < 0 else "#fbbf24"
                    st.markdown(f"""
                    <div class='bear-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span class='bear-ticker'>{r['ticker']}</span>
                            <span style='font-size:0.75rem;'>{r.get('risk_level','')}</span>
                        </div>
                        <div class='bear-meta'>
                            ₹{r['last_close']:,.2f} &nbsp;·&nbsp;
                            <span style='color:{chg_color};'>{chg:+.2f}% (5D)</span>
                            &nbsp;·&nbsp; {rsi_str} &nbsp;·&nbsp; Score: {r['score']}
                        </div>
                        <div class='bear-reason'>⚠️ {reasons}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>")
        st.markdown("""
        <div class='info-box' style='font-size:0.72rem;'>
        🔴 <b>High risk</b> = score ≥ 5 &nbsp;|&nbsp;
        🟡 <b>Medium risk</b> = score 2–4 &nbsp;|&nbsp;
        Scoring uses: 10yr annual returns, RSI, MACD, EMA crossovers, ML prediction, news sentiment.
        <br>TODAY = 1-day return + ML &nbsp;|&nbsp; THIS WEEK = 5-day return + MACD/RSI &nbsp;|&nbsp;
        NEXT WEEK = 20-day trend + 10yr history + news.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 6 — BULLISH SCREENER  (NEW: 10yr + ML + News + horizons)
# ════════════════════════════════════════════════════════════
with tab_bull:
    st.markdown("<div class='sec-header'>🚀 Stocks Likely to Rise — Deep Bullish Screener</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='border-color:rgba(0,212,170,0.25);'>
    This screener scans <b>25 popular NSE stocks</b> using <b>10 years of historical data</b>,
    all technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR, OBV),
    a <b>Random Forest ML signal</b>, and <b>live news sentiment</b>.
    It identifies the <b>top 5 stocks most likely to rise today, this week, and next week</b>.<br><br>
    ⚠️ <b>Not financial advice.</b> Always conduct your own research before investing.
    </div>
    <br>
    """, unsafe_allow_html=True)

    # Controls
    bull_c1, bull_c2, bull_c3 = st.columns([1, 1, 2])
    with bull_c2:
        run_bull_screen = st.button("🚀 Run Bullish Screener", type="primary")

    if run_bull_screen or "deep_bull_results" not in st.session_state:
        with st.spinner("🚀 Deep-scanning 25 NSE stocks (10yr data · ML · News)… ~45s"):
            bull_r, bear_r = load_deep_screen()
            st.session_state["deep_bull_results"] = bull_r
            st.session_state["deep_bear_results"] = bear_r
            st.session_state["deep_results"] = True

    bull_results = st.session_state.get("deep_bull_results", [])

    if not bull_results:
        st.info("ℹ️ No strong bullish signals found right now. Try again later or after market hours.")
    else:
        top5_bull = bull_results[:5]
        st.markdown(f"### 🚀 Top {min(5, len(top5_bull))} Bullish Stocks — Likely to Rise")

        # Time-horizon legend
        st.markdown("""
        <div style='background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.2);
             border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem;'>
            <div style='font-size:0.82rem; color:#00d4aa; font-weight:700; margin-bottom:0.6rem;'>
                🚀 PREDICTED RISE HORIZONS
            </div>
            <div style='display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.8rem; color:#8892a4;'>
                <span><span class='horizon-pill pill-day'>TODAY</span> 1-day momentum + ML signal</span>
                <span><span class='horizon-pill pill-week'>THIS WEEK</span> 5-day return + MACD Bull Cross + RSI oversold</span>
                <span><span class='horizon-pill pill-next'>NEXT WEEK</span> 20-day trend + Golden Cross + 10yr win-rate + news</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Summary bar charts side-by-side ──────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            tickers_bull = [r["ticker"].replace(".NS", "") for r in top5_bull]
            ret5_vals    = [r.get("ret_5d", 0) for r in top5_bull]
            fig_bull_bar = go.Figure(go.Bar(
                x=tickers_bull,
                y=ret5_vals,
                marker_color=["#00d4aa" if v >= 0 else "#ff4d6d" for v in ret5_vals],
                text=[f"{v:+.2f}%" for v in ret5_vals],
                textposition="outside",
                name="5D %",
            ))
            fig_bull_bar.update_layout(
                title="5-Day Return — Top 5 Bullish Picks",
                height=300,
                **{**PLOTLY_LAYOUT, "yaxis": dict(
                    title="5D % Change",
                    showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                )},
            )
            st.plotly_chart(fig_bull_bar, use_container_width=True)

        with chart_col2:
            comp_vals = [r.get("composite", 0) for r in top5_bull]
            fig_comp_bull = go.Figure(go.Bar(
                x=tickers_bull,
                y=comp_vals,
                marker_color=["#00d4aa" if v >= 0 else "#fbbf24" for v in comp_vals],
                text=[f"{v:+.0f}" for v in comp_vals],
                textposition="outside",
                name="Composite",
            ))
            fig_comp_bull.update_layout(
                title="Composite Score (+100 = most bullish)",
                height=300,
                **{**PLOTLY_LAYOUT, "yaxis": dict(
                    title="Score", range=[-110, 110],
                    showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.15)",
                )},
            )
            st.plotly_chart(fig_comp_bull, use_container_width=True)

        # ── 10-year annual return history chart ───────────────
        ml_vals  = [r.get("ml_pct",     0) for r in top5_bull]
        avg_rets = [r.get("avg_yr_ret", 0) for r in top5_bull]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=tickers_bull, y=avg_rets,
            name="10Y Avg Annual Return (%)",
            marker_color="rgba(79,156,249,0.7)",
            text=[f"{v:+.1f}%" for v in avg_rets],
            textposition="outside",
        ))
        fig_hist.add_trace(go.Scatter(
            x=tickers_bull, y=ml_vals,
            name="ML Next-Day Signal (%)",
            mode="markers+lines",
            marker=dict(color="#fbbf24", size=10, symbol="diamond"),
            line=dict(color="#fbbf24", width=2, dash="dot"),
        ))
        fig_hist.update_layout(
            title="10-Year Avg Annual Return vs ML Next-Day Signal",
            height=300,
            barmode="group",
            **{**PLOTLY_LAYOUT, "yaxis": dict(
                title="%",
                showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=True,
                zerolinecolor="rgba(255,255,255,0.12)",
            )},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Detailed cards ────────────────────────────────────
        st.markdown("<div class='sec-header'>Detailed Bullish Signals — Top 5</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for rank, r in enumerate(top5_bull, 1):
            ret1d   = r.get("ret_1d",    0)
            ret5d   = r.get("ret_5d",    0)
            ret20d  = r.get("ret_20d",   0)
            ml_sig  = r.get("ml_pct",    0)
            news_s  = r.get("news_score",0)
            avg_yr  = r.get("avg_yr_ret",0)
            win_r   = r.get("win_rate",  0)
            pats    = ", ".join(r.get("patterns", [])) or "None detected"
            rsi_s   = f"RSI: {r['rsi']:.0f}" if r.get("rsi") is not None else ""
            reasons = " &nbsp;·&nbsp; ".join(r["signal_reasons"])
            vol_r   = r.get("vol_ratio", 1.0)
            vol_q   = r.get("vol_quality", "N/A")
            w_score = r.get("weighted_score", 0)
            w_label = r.get("conviction_label", "")
            conv_reasons = r.get("conviction_reasons", [])
            mkt_reg = r.get("market_regime", "Unknown")
            sec_d   = r.get("sector_data", {})
            rr_d    = r.get("rr_data", {})

            day_color  = "#00d4aa" if ret1d  >= 0 else "#ff4d6d"
            week_color = "#00d4aa" if ret5d  >= 0 else "#ff4d6d"
            mo_color   = "#00d4aa" if ret20d >= 0 else "#ff4d6d"
            ml_color   = "#00d4aa" if ml_sig >= 0 else "#ff4d6d"
            news_color = "#00d4aa" if news_s > 0 else ("#ff4d6d" if news_s < 0 else "#8892a4")
            vol_q_css  = {"Confirmed Rally":"vol-confirmed","Low Volume Rally":"vol-low-rally",
                          "High Vol Decline":"vol-hi-decline"}.get(vol_q,"vol-weak")
            w_color    = "#00d4aa" if w_score >= 60 else ("#4f9cf9" if w_score >= 45 else "#fbbf24")
            risk_pct_b = round(100 - w_score, 0)
            risk_bar_c = "#ff4d6d" if risk_pct_b > 60 else ("#fbbf24" if risk_pct_b > 40 else "#00d4aa")
            rr_txt     = f"R:R {rr_d['rr_ratio']:.1f}× ({rr_d['signal_quality']})" if rr_d and "rr_ratio" in rr_d else ""
            sector_txt = (f"vs {sec_d.get('sector_index','')}: {sec_d.get('outperformance',0):+.1f}%"
                          if sec_d.get("sector_index") else "")

            st.markdown(f"""
            <div class='bull-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-size:0.75rem; color:#8892a4; margin-right:6px;'>#{rank}</span>
                        <span class='bull-ticker'>{r['ticker']}</span>
                    </div>
                    <div style='display:flex; gap:6px; align-items:center;'>
                        <span style='font-size:0.75rem;'>{r.get('confidence','')}</span>
                        <span style='font-size:0.75rem; color:#8892a4;'>Score: {r['score']}</span>
                    </div>
                </div>
                <div class='bull-meta' style='margin-top:0.5rem;'>
                    Last Close: <b>₹{r['last_close']:,.2f}</b>
                    &nbsp;·&nbsp; {rsi_s}
                    &nbsp;·&nbsp; Composite: <b style='color:#00d4aa;'>{r.get('composite', 0):+.0f}</b>
                    &nbsp;·&nbsp; Volatility: {r.get('volatility', 0):.1f}%
                    &nbsp;·&nbsp; <span class='badge {vol_q_css}'>{vol_q}</span>
                    &nbsp;·&nbsp; Vol: {vol_r:.1f}x {" ⚡" if vol_r >= 1.5 else ""}
                    {f"&nbsp;·&nbsp; Regime: <b style='color:" + ("#00d4aa" if mkt_reg=="Bull" else "#ff4d6d") + f";'>{mkt_reg}</b>" if mkt_reg != "Unknown" else ""}
                </div>
                <div style='display:flex; gap:1.5rem; margin-top:0.55rem; flex-wrap:wrap; font-size:0.75rem;'>
                    <span>
                        <span class='horizon-pill pill-day'>TODAY</span>
                        <span style='color:{day_color}; font-weight:600;'>{ret1d:+.2f}%</span>
                        &nbsp; ML: <span style='color:{ml_color};'>{ml_sig:+.1f}%</span>
                    </span>
                    <span>
                        <span class='horizon-pill pill-week'>THIS WEEK</span>
                        <span style='color:{week_color}; font-weight:600;'>{ret5d:+.2f}%</span>
                    </span>
                    <span>
                        <span class='horizon-pill pill-next'>NEXT WEEK</span>
                        <span style='color:{mo_color}; font-weight:600;'>{ret20d:+.2f}%</span>
                        &nbsp; News: <span style='color:{news_color};'>{news_s:+.1f}</span>
                    </span>
                </div>
                <div style='font-size:0.72rem; color:#8892a4; margin-top:0.4rem;'>
                    📈 10Y Avg Annual: <b style='color:#00d4aa;'>{avg_yr:+.1f}%</b>
                    &nbsp;·&nbsp; 10Y Win Rate: <b>{win_r:.0f}%</b>
                    &nbsp;·&nbsp; Patterns: {pats}
                    {f"&nbsp;·&nbsp; {rr_txt}" if rr_txt else ""}
                    {f"&nbsp;·&nbsp; Sector {sector_txt}" if sector_txt else ""}
                </div>
                <div class='bull-reason'>✅ {reasons}</div>
                <!-- Conviction Summary -->
                <div style='margin-top:0.55rem; background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.15);
                     border-radius:8px; padding:0.5rem 0.8rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;'>
                        <div>
                            <div style='font-size:0.65rem; color:#4f9cf9; font-weight:700; letter-spacing:0.04em;'>🎯 CONVICTION SUMMARY</div>
                            <div style='font-size:0.9rem; font-weight:700; color:{w_color};'>{w_score:.0f}/100 — {w_label}</div>
                        </div>
                        <div style='min-width:120px;'>
                            <div style='font-size:0.62rem; color:#8892a4; margin-bottom:2px;'>Risk Meter: {risk_pct_b:.0f}/100</div>
                            <div style='background:rgba(255,255,255,0.06); border-radius:6px; height:6px; overflow:hidden;'>
                                <div style='height:100%; border-radius:6px; background:{risk_bar_c}; width:{risk_pct_b}%;'></div>
                            </div>
                        </div>
                    </div>
                    <div style='font-size:0.68rem; color:#8892a4; margin-top:0.25rem; line-height:1.5;'>
                        {" · ".join(conv_reasons) if conv_reasons else "No strong conviction drivers detected."}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(bull_results) > 5:
            with st.expander(f"Show remaining {len(bull_results) - 5} bullish stocks"):
                for r in bull_results[5:]:
                    rsi_str = f"RSI: {r['rsi']:.0f}" if r.get("rsi") is not None else ""
                    reasons = " &nbsp;·&nbsp; ".join(r["signal_reasons"])
                    chg     = r.get("ret_5d", 0)
                    cc      = "#00d4aa" if chg >= 0 else "#ff4d6d"
                    st.markdown(f"""
                    <div class='bull-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span class='bull-ticker'>{r['ticker']}</span>
                            <span style='font-size:0.75rem;'>{r.get('confidence','')}</span>
                        </div>
                        <div class='bull-meta'>
                            ₹{r['last_close']:,.2f} &nbsp;·&nbsp;
                            <span style='color:{cc};'>{chg:+.2f}% (5D)</span>
                            &nbsp;·&nbsp; {rsi_str} &nbsp;·&nbsp; Score: {r['score']}
                        </div>
                        <div class='bull-reason'>✅ {reasons}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>")
        st.markdown("""
        <div class='info-box' style='font-size:0.72rem; border-color:rgba(0,212,170,0.2);'>
        🟢 <b>Strong</b> = bull score ≥ 6 &nbsp;|&nbsp;
        🟡 <b>Moderate</b> = bull score 3–5 &nbsp;|&nbsp;
        Composite score: −100 (most bearish) to +100 (most bullish).<br>
        TODAY = 1-day return + ML next-day signal &nbsp;|&nbsp;
        THIS WEEK = 5-day return + MACD Bull Cross + RSI &nbsp;|&nbsp;
        NEXT WEEK = 20-day trend + Golden Cross + 10yr win-rate + news sentiment.
        </div>
        """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════
#  TAB 7 — RISK & VOLUME  (NEW)
# ════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown("<div class='sec-header'>⚖️ Risk Management & Volume Intelligence</div>",
                unsafe_allow_html=True)

    # ── Market Regime Banner ──────────────────────────────────
    with st.spinner("🌐 Fetching Nifty 50 market regime…"):
        regime_data = fetch_market_regime()

    regime       = regime_data.get("regime", "Unknown")
    nifty_px     = regime_data.get("nifty_price")
    nifty_ema200 = regime_data.get("nifty_ema200")
    pct_ema200   = regime_data.get("pct_from_ema200")
    mkt_mult     = regime_data.get("multiplier", 1.0)

    if regime == "Bull":
        regime_color = "#00d4aa"
        regime_icon  = "🟢"
        regime_msg   = "Nifty 50 is above its 200-day EMA — favourable broad market conditions."
    elif regime == "Bear":
        regime_color = "#ff4d6d"
        regime_icon  = "🔴"
        regime_msg   = "⚠️ Nifty 50 is BELOW its 200-day EMA — apply caution. Buy signals are discounted by 30%."
    else:
        regime_color = "#fbbf24"
        regime_icon  = "🟡"
        regime_msg   = "Market regime data unavailable."

    st.markdown(f"""
    <div style='background:rgba({("0,212,170" if regime=="Bull" else "255,77,109" if regime=="Bear" else "251,191,36")},0.07);
         border:1px solid rgba({("0,212,170" if regime=="Bull" else "255,77,109" if regime=="Bear" else "251,191,36")},0.25);
         border-radius:10px; padding:1rem 1.4rem; margin-bottom:1rem;'>
        <div style='font-size:1rem; font-weight:700; color:{regime_color}; margin-bottom:0.3rem;'>
            {regime_icon} GLOBAL MARKET REGIME: {regime.upper()} MARKET
        </div>
        <div style='font-size:0.82rem; color:#8892a4;'>{regime_msg}</div>
        {'<div style="font-size:0.78rem; color:#8892a4; margin-top:0.4rem;">Nifty 50: ₹' + f"{nifty_px:,.0f}" + ' &nbsp;|&nbsp; 200-EMA: ₹' + f"{nifty_ema200:,.0f}" + ' &nbsp;|&nbsp; Distance: <b style="color:' + regime_color + '">' + f"{pct_ema200:+.2f}%" + '</b></div>' if nifty_px else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── Sector Performance ────────────────────────────────────
    st.markdown("<div class='sec-header'>🏭 Sector Correlation Analysis</div>",
                unsafe_allow_html=True)

    with st.spinner("📡 Fetching sector performance…"):
        sector_info = fetch_sector_performance(ticker)

    sec_idx  = sector_info.get("sector_index")
    stk_ret  = sector_info.get("stock_ret5", 0)
    sec_ret  = sector_info.get("sector_ret5", 0)
    out_perf = sector_info.get("outperformance", 0)
    s_boost  = sector_info.get("sector_boost", 0)

    if sec_idx:
        sp_color  = "#00d4aa" if out_perf > 0 else "#ff4d6d"
        sp_label  = "Outperforming 🚀" if out_perf > 0 else "Underperforming ⚠️"
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Stock 5D Return",    f"{stk_ret:+.2f}%")
        sc2.metric("Sector 5D Return",   f"{sec_ret:+.2f}%",
                   delta=f"vs {sec_idx}")
        sc3.metric("Outperformance",     f"{out_perf:+.2f}%",
                   delta=sp_label)
        sc4.metric("Sector Boost Score", f"{s_boost}/2")

        fig_sec = go.Figure(go.Bar(
            x=["This Stock", "Sector Index"],
            y=[stk_ret, sec_ret],
            marker_color=["#00d4aa" if stk_ret >= sec_ret else "#fbbf24", "#4f9cf9"],
            text=[f"{stk_ret:+.2f}%", f"{sec_ret:+.2f}%"],
            textposition="outside",
        ))
        fig_sec.update_layout(
            title=f"5-Day Return: {ticker} vs {sec_idx}",
            height=280, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info(f"ℹ️ Sector mapping not available for {ticker}. Supported: NSE Bank, IT, Pharma, Auto, FMCG, Energy.")

    # ── Volume Profile ────────────────────────────────────────
    st.markdown("<div class='sec-header'>📊 Volume Profile & Smart Money Detection</div>",
                unsafe_allow_html=True)

    # Enrich current df with volume profile if not already done
    df_vol = df.copy()
    if "Vol_SMA20" not in df_vol.columns:
        df_vol = add_volume_profile(df_vol)

    last_vol_row = df_vol.iloc[-1]
    vol_ratio_now = float(last_vol_row.get("Vol_Surge", 1.0)) if "Vol_Surge" in df_vol.columns else 1.0
    vol_sma20_now = float(last_vol_row.get("Vol_SMA20", 0)) if "Vol_SMA20" in df_vol.columns else 0
    vol_quality_now = str(last_vol_row.get("Vol_Quality", "N/A")) if "Vol_Quality" in df_vol.columns else "N/A"

    vq_color = {"Confirmed Rally": "#00d4aa", "Low Volume Rally": "#fbbf24",
                "High Vol Decline": "#ff4d6d", "Weak Decline": "#8892a4"}.get(vol_quality_now, "#8892a4")

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Volume Surge Ratio",  f"{vol_ratio_now:.2f}x",
               delta="SURGE" if vol_ratio_now >= 1.5 else "Normal")
    vc2.metric("20-Day Avg Volume",   f"{int(vol_sma20_now):,}" if vol_sma20_now else "N/A")
    vc3.metric("Volume Quality",      vol_quality_now)

    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
         border-radius:8px; padding:0.75rem 1rem; font-size:0.82rem; margin:0.5rem 0 1rem;'>
        {"🚀 <b style='color:#00d4aa;'>VOLUME BREAKOUT DETECTED</b> — Volume is " + f"{vol_ratio_now:.1f}x" + " the 20-day average. This signals <b>institutional/smart money activity</b>. Combined with price action, this indicates a <b>conviction move</b>." if vol_ratio_now >= 1.5 else
         "⚠️ <b style='color:#fbbf24;'>LOW VOLUME MOVE</b> — Price action is not confirmed by volume. Treat with caution — this could be a <b>weak rally</b> susceptible to reversal." if 1.0 < vol_ratio_now < 1.5 else
         "ℹ️ Volume is <b>normal</b> — no surge or institutional signal detected."}
    </div>
    """, unsafe_allow_html=True)

    # Volume chart with SMA overlay
    if "Vol_SMA20" in df_vol.columns:
        vol_colors_rr = ["#00d4aa" if c >= o else "#ff4d6d"
                         for c, o in zip(df_vol["Close"], df_vol["Open"])]
        fig_vol_rr = go.Figure()
        fig_vol_rr.add_trace(go.Bar(
            x=df_vol.index, y=df_vol["Volume"],
            marker_color=vol_colors_rr, name="Volume", opacity=0.65,
        ))
        fig_vol_rr.add_trace(go.Scatter(
            x=df_vol.index, y=df_vol["Vol_SMA20"],
            name="20D Avg Volume", line=dict(color="#fbbf24", width=2),
        ))
        # Highlight surge bars
        surge_mask = df_vol["Vol_Surge"] >= 1.5 if "Vol_Surge" in df_vol.columns else pd.Series(False, index=df_vol.index)
        surge_df   = df_vol[surge_mask]
        if not surge_df.empty:
            fig_vol_rr.add_trace(go.Bar(
                x=surge_df.index, y=surge_df["Volume"],
                name="Volume Surge (≥1.5x)", marker_color="rgba(255,215,0,0.55)", opacity=0.9,
            ))
        fig_vol_rr.update_layout(
            title="Volume vs 20-Day SMA — Surge Highlighted in Gold",
            height=280, barmode="overlay", **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_vol_rr, use_container_width=True)

    # ── ATR Risk-Reward Framework ─────────────────────────────
    st.markdown("<div class='sec-header'>⚖️ Dynamic Risk-Reward Framework (ATR-Based)</div>",
                unsafe_allow_html=True)

    df_rr = df.copy()
    if "ATR" not in df_rr.columns:
        from data_engine import add_atr as _add_atr
        df_rr = _add_atr(df_rr)

    rr = compute_risk_reward(df_rr)

    if "error" in rr:
        st.warning(f"Risk-Reward calculation unavailable: {rr['error']}")
    else:
        rr_color = {"Strong Buy": "#00d4aa", "Buy": "#4f9cf9", "Neutral": "#8892a4"}[rr["signal_quality"]]
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Entry (Current Price)",   f"{currency} {rr['entry_price']:,.2f}")
        rc2.metric("Stop-Loss (1.5×ATR)",     f"{currency} {rr['stop_loss']:,.2f}",
                   delta=f"Risk: {currency} {rr['risk_per_share']:,.2f}", delta_color="inverse")
        rc3.metric("Target 2:1 R:R",          f"{currency} {rr['target_2r']:,.2f}")
        rc4.metric("R:R Signal",              rr["signal_quality"],
                   delta=f"R:R = {rr['rr_ratio']:.1f}×")

        st.markdown(f"""
        <div class='rr-card'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;'>
                <div>
                    <div class='rr-label'>SIGNAL QUALITY</div>
                    <div class='rr-value' style='color:{rr_color}; font-size:1.3rem;'>
                        {'🎯' if rr["signal_quality"]=="Strong Buy" else '✅' if rr["signal_quality"]=="Buy" else '⚪'} {rr["signal_quality"]}
                    </div>
                </div>
                <div>
                    <div class='rr-label'>ATR Value</div>
                    <div class='rr-value'>{currency} {rr["atr_value"]:,.2f}</div>
                </div>
                <div>
                    <div class='rr-label'>Stop-Loss</div>
                    <div class='rr-value' style='color:#ff4d6d;'>{currency} {rr["stop_loss"]:,.2f}</div>
                </div>
                <div>
                    <div class='rr-label'>Target 1:1</div>
                    <div class='rr-value' style='color:#8892a4;'>{currency} {rr["target_1r"]:,.2f}</div>
                </div>
                <div>
                    <div class='rr-label'>Target 2:1 ✅</div>
                    <div class='rr-value' style='color:#4f9cf9;'>{currency} {rr["target_2r"]:,.2f}</div>
                </div>
                <div>
                    <div class='rr-label'>Target 3:1 🚀</div>
                    <div class='rr-value' style='color:#00d4aa;'>{currency} {rr["target_3r"]:,.2f}</div>
                </div>
            </div>
            <div style='font-size:0.74rem; color:#8892a4; margin-top:0.7rem;'>
                🔑 A signal is flagged <b>Strong Buy</b> only when projected upside ≥ 2× the calculated risk
                (1.5×ATR stop-loss). Current R:R = <b style='color:{rr_color};'>{rr["rr_ratio"]:.1f}×</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Weighted Composite Score / Risk Meter ─────────────────
    st.markdown("<div class='sec-header'>🎯 Weighted Conviction Score & Risk Meter</div>",
                unsafe_allow_html=True)

    # Compute on-the-fly for active ticker
    if "RSI" in df.columns and len(df) >= 30:
        last_r   = df.iloc[-1]
        rsi_now  = float(last_r.get("RSI", 50))
        macd_h   = float(last_r.get("MACD_Hist", 0)) if "MACD_Hist" in df.columns else 0
        ema20_v  = float(last_r.get("EMA_20", cur_px)) if "EMA_20" in df.columns else cur_px
        ema50_v  = float(last_r.get("EMA_50", cur_px)) if "EMA_50" in df.columns else cur_px

        # Technical sub-score (rough, 0–100)
        ts = 50.0
        ts += 20 if cur_px > ema20_v else -20
        ts += 15 if cur_px > ema50_v else -15
        ts += 10 if rsi_now < 50 else -10
        ts += 5  if macd_h > 0 else -5
        ts = float(np.clip(ts, 0, 100))

        wc = compute_weighted_composite(
            technical_score    = ts,
            historical_win_rate= 55.0,     # fallback; full scan fetches 10yr
            news_score         = 0.0,
            volume_surge_ratio = vol_ratio_now,
            sector_boost       = s_boost,
            market_multiplier  = mkt_mult,
        )
        ws     = wc["weighted_score"]
        wlabel = wc["conviction_label"]
        wreasons = wc["reasons"]

        # Color gradient: red→amber→green
        if ws >= 65:
            bar_color = "#00d4aa"
        elif ws >= 45:
            bar_color = "#4f9cf9"
        elif ws >= 30:
            bar_color = "#fbbf24"
        else:
            bar_color = "#ff4d6d"

        # Risk Meter (inverted — high conviction = low risk)
        risk_pct = round(100 - ws, 1)

        wc1, wc2 = st.columns([1, 1])
        with wc1:
            st.markdown(f"""
            <div class='conviction-box'>
                <div class='conviction-title'>📊 Weighted Conviction Score</div>
                <div class='conviction-score'>{ws:.1f}<span style='font-size:1rem; color:#8892a4;'>/100</span></div>
                <div class='conviction-label'>{wlabel}</div>
                <div style='background:rgba(255,255,255,0.06); border-radius:8px; height:10px; margin:0.5rem 0 0.3rem; overflow:hidden;'>
                    <div style='height:100%; border-radius:8px; background:{bar_color}; width:{ws}%;'></div>
                </div>
                <div class='conviction-reasons'>
                    {'<br>'.join("✓ " + r for r in wreasons) if wreasons else "No strong signals detected."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with wc2:
            st.markdown(f"""
            <div class='risk-meter-wrap'>
                <div style='font-size:0.78rem; font-weight:700; color:#ff4d6d; letter-spacing:0.05em;
                     text-transform:uppercase; margin-bottom:0.2rem;'>🔴 Risk Meter</div>
                <div style='font-size:1.8rem; font-weight:700; color:{"#ff4d6d" if risk_pct>60 else "#fbbf24" if risk_pct>40 else "#00d4aa"};'>
                    {risk_pct:.0f}<span style='font-size:0.9rem; color:#8892a4;'>/100</span>
                </div>
                <div class='risk-bar-bg'>
                    <div class='risk-bar-fill' style='width:{risk_pct}%;
                         background:{"#ff4d6d" if risk_pct>60 else "#fbbf24" if risk_pct>40 else "#00d4aa"};'>
                    </div>
                </div>
                <div style='font-size:0.74rem; color:#8892a4;'>
                    {"🔴 <b>High Risk</b> — most signals are bearish or weak" if risk_pct > 60 else
                     "🟡 <b>Medium Risk</b> — mixed signals, proceed with caution" if risk_pct > 40 else
                     "🟢 <b>Low Risk</b> — strong conviction, favourable setup"}
                </div>
                <div style='font-size:0.72rem; color:#8892a4; margin-top:0.5rem; line-height:1.5;'>
                    <b>Score Breakdown:</b><br>
                    Technical: {wc["breakdown"].get("technical", 0):.1f}
                    &nbsp;·&nbsp; History: {wc["breakdown"].get("historical", 0):.1f}
                    &nbsp;·&nbsp; News: {wc["breakdown"].get("news", 0):.1f}
                    &nbsp;·&nbsp; Volume: {wc["breakdown"].get("volume", 0):.1f}
                    &nbsp;·&nbsp; Sector: {wc["breakdown"].get("sector", 0):.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Breakdown radar chart ─────────────────────────────
        breakdown_labels = ["Technical (30%)", "Historical (20%)", "News (20%)", "Volume (15%)", "Sector (15%)"]
        breakdown_values = [
            wc["breakdown"].get("technical",  0) / 30 * 100,
            wc["breakdown"].get("historical", 0) / 20 * 100,
            wc["breakdown"].get("news",       0) / 20 * 100,
            wc["breakdown"].get("volume",     0) / 15 * 100,
            wc["breakdown"].get("sector",     0) / 15 * 100,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r     = breakdown_values + [breakdown_values[0]],
            theta = breakdown_labels + [breakdown_labels[0]],
            fill  = "toself",
            fillcolor = "rgba(0,212,170,0.1)",
            line  = dict(color="#00d4aa", width=2),
            name  = "Score Breakdown",
        ))
        fig_radar.update_layout(
            polar = dict(
                bgcolor       = "rgba(0,0,0,0)",
                radialaxis    = dict(visible=True, range=[0,100],
                                     gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=9, color="#8892a4")),
                angularaxis   = dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=9, color="#f0f4ff")),
            ),
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="#f0f4ff"),
            height        = 340,
            showlegend    = False,
            title         = dict(text="Conviction Score Breakdown by Component", font=dict(size=12)),
            margin        = dict(l=40, r=40, t=50, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("Enable RSI / MACD in the sidebar to see the full conviction breakdown.")

    st.markdown("""
    <div class='info-box' style='font-size:0.74rem; margin-top:0.5rem;'>
    ⚖️ <b>Weights:</b> Technical 30% · Historical Win-Rate 20% · News Sentiment 20% · Volume Strength 15% · Sector/Market 15%<br>
    🔴 <b>Risk Meter:</b> Inverse of Conviction Score. High conviction = low risk. &nbsp;|&nbsp;
    ⚠️ In Bear market regime (Nifty below 200-EMA), all bullish buy signals are discounted by 30%.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 8 — AI ANALYST CHAT
# ════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("<div class='sec-header'>💬 AI Stock Analyst — Deep Market Intelligence</div>",
                unsafe_allow_html=True)

    # ── API key loaded from .env (GEMINI_API_KEY) — no UI input needed ──
    gemini_key = os.getenv("GEMINI_API_KEY", "")


    # ── HOW IT WORKS ──────────────────────────────────────────
    with st.expander("🧠 How the AI prediction engine works"):
        st.markdown("""
        The AI analyst runs a **7-layer deep analysis** on every stock before answering:

        1. **10-Year Price History** — annual returns, best/worst years, compound growth
        2. **Technical Indicators** — RSI, MACD, EMA (20/50), Bollinger Bands, Stochastic, ATR
        3. **Pattern Recognition** — Golden Cross, Death Cross, MACD crossover, BB Squeeze, OBV divergence
        4. **Momentum Scoring** — 1D, 5D, 20D, 60D, 1Y momentum weighted composite
        5. **Volume Analysis** — OBV trend, volume surge detection, institutional activity signals
        6. **News Sentiment** — recent headlines scored for bullish/bearish language
        7. **Composite Score** — all factors merged into a single −100 to +100 score with confidence level

        The AI then reasons over all this data to rank stocks and give predictions.

        > ⚠️ No prediction system is 100% accurate. Markets can be unpredictable.
        > Always use this as one input among many — not as sole financial advice.
        """)

    st.markdown("---")

    # ── QUICK QUESTIONS ───────────────────────────────────────
    st.markdown("**⚡ Quick Questions — click to ask instantly:**")
    quick_questions = [
        ("🚀 Top 5 Gainers",      "Which 5 NSE stocks from the watchlist are most likely to gain in the next few days? Analyse 10 years of data, all technical indicators, patterns and news sentiment. Rank them with confidence scores."),
        ("📉 Top 5 Losers",       "Which 5 NSE stocks from the watchlist are most likely to fall or lose value in the next few days? Use 10-year history, RSI, MACD, patterns and sentiment to rank them with reasons."),
        ("📊 Current Stock",      f"Give me a complete deep analysis of {ticker}: 10-year history, current technical signals, patterns detected, composite score, and your prediction for the next 3-5 days with confidence level."),
        ("⚖️ Buy or Sell?",       f"Based on 10 years of data and all technical indicators, should I buy, sell or hold {ticker} right now? Give a clear verdict with supporting evidence."),
        ("🏆 Best Long-Term",     "Which 3 stocks from the watchlist have the strongest long-term growth based on 10-year returns, low drawdown and current bullish setup? Explain your reasoning."),
        ("⚡ Breakout Stocks",    "Which NSE stocks are showing a Bollinger Band squeeze or Golden Cross pattern right now? These often precede big moves — identify them and predict direction."),
    ]

    qrow1 = st.columns(3)
    qrow2 = st.columns(3)
    all_cols = qrow1 + qrow2
    for i, (label, full_q) in enumerate(quick_questions):
        if all_cols[i].button(label, key=f"qq_{i}", use_container_width=True):
            if not gemini_key:
                st.error("❌ GEMINI_API_KEY not found in .env file. Please add it and restart the app.")
            else:
                st.session_state["pending_question"] = full_q

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EXTRA TICKERS ─────────────────────────────────────────
    with st.expander("🔧 Advanced: Add extra tickers to analyse"):
        extra_raw = st.text_input(
            "Extra tickers (comma-separated)",
            placeholder="e.g. TCS.NS, INFY.NS, HDFCBANK.NS",
            key="extra_tickers_input",
        )
        extra_tickers_chat = [t.strip().upper() for t in extra_raw.split(",") if t.strip()] if extra_raw else []
        if extra_tickers_chat:
            st.caption(f"Will also analyse: {', '.join(extra_tickers_chat)}")

    # ── CHAT HISTORY ──────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.markdown("<div class='sec-header'>Conversation</div>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-msg-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Render markdown properly for AI responses
                with st.container():
                    st.markdown(
                        "<div style='border-left:3px solid #00d4aa; padding-left:1rem; margin:0.5rem 0;'>"
                        "<span style='font-size:0.65rem; color:#00d4aa; font-weight:700; letter-spacing:0.05em;'>🤖 STOCKIQ AI</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── INPUT FORM ────────────────────────────────────────────
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question",
            placeholder="e.g. Which 5 stocks will rise this week? / Analyse RELIANCE.NS in detail / Compare TCS vs INFY / Is the market bullish right now?",
            height=90,
            label_visibility="collapsed",
        )
        fcol1, fcol2, fcol3 = st.columns([5, 1, 1])
        send_btn  = fcol1.form_submit_button("📤 Send to AI Analyst", type="primary", use_container_width=True)
        clear_btn = fcol3.form_submit_button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    # ── PROCESS QUESTION ──────────────────────────────────────
    pending  = st.session_state.pop("pending_question", None)
    question = pending or (user_input.strip() if send_btn and user_input.strip() else None)

    if question:
        if not gemini_key:
            st.error("❌ GEMINI_API_KEY not set in .env file. Add GEMINI_API_KEY=AIza... and restart Streamlit.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})

            # Animated status — purely cosmetic, runs before the blocking call
            status_box = st.empty()
            prog       = st.progress(0)

            def show_status(pct, msg):
                prog.progress(pct)
                status_box.markdown(
                    f"<div style='background:rgba(79,156,249,0.08);border:1px solid rgba(79,156,249,0.2);"
                    f"border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#8892a4;'>{msg}</div>",
                    unsafe_allow_html=True,
                )

            show_status(10, "📡 Fetching 10-year historical data for each stock...")
            show_status(30, "🔬 Computing RSI · MACD · Bollinger Bands · Stochastic...")
            show_status(55, "🧩 Detecting patterns · Scoring news sentiment...")
            show_status(75, "⚗️ Running composite scoring model...")
            show_status(90, "🤖 Sending analysis to Gemini AI — generating prediction...")

            # ── Blocking call (no thread — Streamlit-safe) ─────────
            answer = ask_ai_analyst(
                user_question=question,
                active_ticker=ticker,
                extra_tickers=extra_tickers_chat if extra_tickers_chat else None,
                news_api_key=NEWS_API_KEY,
                gemini_api_key=gemini_key,
            )

            prog.progress(100)
            status_box.empty()
            prog.empty()

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    # ── EMPTY STATE ───────────────────────────────────────────
    if not st.session_state.chat_history and not question:
        st.markdown("""
        <div style='text-align:center; padding:2.5rem 1rem; color:#8892a4;'>
            <div style='font-size:3rem; margin-bottom:0.6rem;'>🤖</div>
            <div style='font-size:1rem; font-weight:600; color:#f0f4ff;'>StockIQ AI Analyst</div>
            <div style='font-size:0.82rem; margin-top:0.4rem; line-height:1.7;'>
                Powered by Claude + 10 years of real market data<br>
                Enter your API key → click a Quick Question or type below
            </div>
            <div style='margin-top:1.2rem; display:flex; justify-content:center; gap:1.5rem; font-size:0.75rem;'>
                <span>📊 10-Year History</span>
                <span>🔬 7 Indicators</span>
                <span>🧩 Pattern Detection</span>
                <span>📰 News Sentiment</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

#  FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("""
<br>
<div style='text-align:center; font-size:0.72rem; color:#8892a4; padding:1rem 0;
     border-top:1px solid rgba(255,255,255,0.06);'>
    StockIQ — AI Market Intelligence Suite &nbsp;|&nbsp;
    Built with Streamlit · yfinance · scikit-learn · Plotly &nbsp;|&nbsp;
    ⚠️ For educational purposes only — not financial advice
</div>
""", unsafe_allow_html=True)
