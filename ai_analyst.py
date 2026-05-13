# ============================================================
# ai_analyst.py
# Ultra-advanced AI Stock Analyst:
#   - 10-year historical data analysis
#   - Technical indicators (RSI, MACD, EMA, Bollinger Bands)
#   - ML-based price prediction signals
#   - News sentiment scoring
#   - Pattern recognition (Golden Cross, Death Cross, etc.)
#   - Volume analysis & momentum scoring
#   - Multi-factor composite scoring for prediction
# ============================================================

import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from data_engine import fetch_stock_data, enrich_data, WATCHLIST
from news_service import score_sentiment

# ── Google Gemini API key ────────────────────────────────────
# Set GEMINI_API_KEY in your .env file, or pass it directly
# from the UI (recommended — no file editing needed).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ════════════════════════════════════════════════════════════
#  ADVANCED INDICATOR CALCULATIONS
# ════════════════════════════════════════════════════════════

def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) to DataFrame."""
    df["BB_Mid"]   = df["Close"].rolling(period).mean()
    df["BB_Std"]   = df["Close"].rolling(period).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator %K and %D."""
    low_min  = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(d).mean()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (volatility measure)."""
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) — cumulative volume flow."""
    direction = np.sign(df["Close"].diff())
    df["OBV"]  = (direction * df["Volume"]).cumsum()
    return df


def add_vwap_approx(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add approximate rolling VWAP."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical * df["Volume"]).rolling(window).sum() / df["Volume"].rolling(window).sum()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator in one shot."""
    df = enrich_data(df, show_ema=True, show_rsi=True, show_macd=True)
    df = add_bollinger_bands(df)
    df = add_stochastic(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_vwap_approx(df)
    return df


# ════════════════════════════════════════════════════════════
#  PATTERN RECOGNITION
# ════════════════════════════════════════════════════════════

def detect_patterns(df: pd.DataFrame) -> list:
    """
    Detect key chart patterns and return a list of signal strings.
    Patterns: Golden Cross, Death Cross, RSI Divergence,
              Bollinger Squeeze, MACD Crossover, OBV Breakout.
    """
    patterns = []
    if len(df) < 55:
        return patterns

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev3 = df.iloc[-4] if len(df) >= 4 else prev

    # Golden Cross: EMA20 crosses above EMA50
    if ("EMA_20" in df.columns and "EMA_50" in df.columns):
        if prev["EMA_20"] < prev["EMA_50"] and last["EMA_20"] > last["EMA_50"]:
            patterns.append("🟢 GOLDEN CROSS detected (EMA20 > EMA50) — strong bullish signal")
        elif prev["EMA_20"] > prev["EMA_50"] and last["EMA_20"] < last["EMA_50"]:
            patterns.append("🔴 DEATH CROSS detected (EMA20 < EMA50) — strong bearish signal")

    # MACD Bullish / Bearish crossover
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        if prev["MACD"] < prev["MACD_Signal"] and last["MACD"] > last["MACD_Signal"]:
            patterns.append("🟢 MACD BULLISH CROSSOVER — momentum turning positive")
        elif prev["MACD"] > prev["MACD_Signal"] and last["MACD"] < last["MACD_Signal"]:
            patterns.append("🔴 MACD BEARISH CROSSOVER — momentum turning negative")

    # Bollinger Band Squeeze (low volatility before breakout)
    if "BB_Width" in df.columns:
        recent_bw = df["BB_Width"].iloc[-20:]
        if last["BB_Width"] < recent_bw.quantile(0.15):
            patterns.append("⚡ BOLLINGER SQUEEZE — low volatility, breakout imminent")
        if last["BB_Pct"] > 0.95:
            patterns.append("🔴 Price at UPPER BOLLINGER BAND — overbought zone")
        elif last["BB_Pct"] < 0.05:
            patterns.append("🟢 Price at LOWER BOLLINGER BAND — oversold / bounce zone")

    # RSI Divergence (price makes new high but RSI doesn't)
    if "RSI" in df.columns:
        rsi_now = last["RSI"]
        if rsi_now >= 70:
            patterns.append(f"🔴 RSI OVERBOUGHT ({rsi_now:.1f}) — potential reversal down")
        elif rsi_now <= 30:
            patterns.append(f"🟢 RSI OVERSOLD ({rsi_now:.1f}) — potential reversal up")
        elif 45 <= rsi_now <= 55:
            patterns.append(f"⚪ RSI NEUTRAL ({rsi_now:.1f}) — no strong signal")

    # Stochastic overbought/oversold
    if "Stoch_K" in df.columns:
        sk = last["Stoch_K"]
        sd = last["Stoch_D"]
        if sk > 80 and sd > 80:
            patterns.append(f"🔴 STOCHASTIC OVERBOUGHT (K={sk:.0f}, D={sd:.0f})")
        elif sk < 20 and sd < 20:
            patterns.append(f"🟢 STOCHASTIC OVERSOLD (K={sk:.0f}, D={sd:.0f}) — bounce likely")

    # OBV trend (volume confirms price)
    if "OBV" in df.columns and len(df) >= 10:
        obv_trend = df["OBV"].iloc[-5:].mean() - df["OBV"].iloc[-10:-5].mean()
        price_trend = df["Close"].iloc[-5:].mean() - df["Close"].iloc[-10:-5].mean()
        if obv_trend > 0 and price_trend > 0:
            patterns.append("🟢 OBV CONFIRMING UPTREND — buyers in control")
        elif obv_trend < 0 and price_trend < 0:
            patterns.append("🔴 OBV CONFIRMING DOWNTREND — sellers in control")
        elif obv_trend > 0 and price_trend < 0:
            patterns.append("🟢 OBV BULLISH DIVERGENCE — price may reverse up (accumulation)")
        elif obv_trend < 0 and price_trend > 0:
            patterns.append("🔴 OBV BEARISH DIVERGENCE — price may reverse down (distribution)")

    # Volume surge
    if len(df) >= 20:
        avg_vol = df["Volume"].iloc[-20:].mean()
        last_vol = last["Volume"]
        if last_vol > avg_vol * 2:
            patterns.append(f"📈 VOLUME SURGE ({last_vol/avg_vol:.1f}x avg) — institutional activity")

    return patterns


# ════════════════════════════════════════════════════════════
#  COMPOSITE SCORE ENGINE
# ════════════════════════════════════════════════════════════

def compute_composite_score(df: pd.DataFrame, patterns: list, news_sentiment: float = 0) -> dict:
    """
    Compute a multi-factor composite score (−100 to +100).
    Positive = bullish, negative = bearish.

    Factors:
      - Technical momentum  (30%)
      - Trend strength      (25%)
      - Oscillator signals  (20%)
      - Pattern signals     (15%)
      - News sentiment      (10%)
    """
    score      = 0.0
    max_score  = 100.0
    breakdown  = {}

    if df.empty or len(df) < 30:
        return {"total": 0, "outlook": "Insufficient Data", "confidence": "Low", "breakdown": {}}

    last = df.iloc[-1]

    # ── 1. Momentum (30 pts) ──────────────────────────────────
    mom = 0
    close = df["Close"]
    def ret(n): return ((close.iloc[-1] - close.iloc[-n-1]) / close.iloc[-n-1]) * 100 if len(close) > n else 0

    r1  = ret(1);   mom += np.clip(r1  * 2,  -6, 6)
    r5  = ret(5);   mom += np.clip(r5  * 1.5,-7, 7)
    r20 = ret(20);  mom += np.clip(r20 * 0.8,-8, 8)
    r60 = ret(60);  mom += np.clip(r60 * 0.3,-9, 9)
    breakdown["momentum"] = round(mom, 1)
    score += mom

    # ── 2. Trend Strength (25 pts) ───────────────────────────
    trend = 0
    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        px    = float(last["Close"])
        e20   = float(last["EMA_20"])
        e50   = float(last["EMA_50"])
        if px > e20: trend += 8
        if px > e50: trend += 7
        if e20 > e50: trend += 10
        if px < e20: trend -= 8
        if px < e50: trend -= 7
        if e20 < e50: trend -= 10
    breakdown["trend"] = round(trend, 1)
    score += trend

    # ── 3. Oscillators (20 pts) ──────────────────────────────
    osc = 0
    if "RSI" in df.columns:
        rsi = float(last["RSI"])
        if rsi < 30:   osc += 10
        elif rsi < 45: osc += 5
        elif rsi > 70: osc -= 10
        elif rsi > 55: osc -= 3

    if "MACD_Hist" in df.columns:
        h = float(last["MACD_Hist"])
        osc += np.clip(h * 5, -10, 10)

    if "Stoch_K" in df.columns:
        sk = float(last["Stoch_K"])
        if sk < 20: osc += 5
        elif sk > 80: osc -= 5
    breakdown["oscillators"] = round(osc, 1)
    score += osc

    # ── 4. Pattern Signals (15 pts) ──────────────────────────
    pat_score = 0
    for p in patterns:
        if "🟢" in p:  pat_score += 3
        elif "🔴" in p: pat_score -= 3
        elif "⚡" in p:  pat_score += 1
    pat_score = np.clip(pat_score, -15, 15)
    breakdown["patterns"] = round(pat_score, 1)
    score += pat_score

    # ── 5. News Sentiment (10 pts) ───────────────────────────
    news_contribution = np.clip(news_sentiment * 5, -10, 10)
    breakdown["news_sentiment"] = round(news_contribution, 1)
    score += news_contribution

    # ── Final Score ───────────────────────────────────────────
    total = round(np.clip(score, -100, 100), 1)

    if total >= 40:
        outlook    = "Strongly Bullish 📈"
        confidence = "High"
    elif total >= 15:
        outlook    = "Mildly Bullish 🟢"
        confidence = "Medium"
    elif total <= -40:
        outlook    = "Strongly Bearish 📉"
        confidence = "High"
    elif total <= -15:
        outlook    = "Mildly Bearish 🔴"
        confidence = "Medium"
    else:
        outlook    = "Neutral / Sideways ⚪"
        confidence = "Low"

    return {
        "total":      total,
        "outlook":    outlook,
        "confidence": confidence,
        "breakdown":  breakdown,
    }


# ════════════════════════════════════════════════════════════
#  NEWS SENTIMENT FOR A TICKER (no API key needed)
# ════════════════════════════════════════════════════════════

def get_news_sentiment_score(ticker: str, news_api_key: str = "") -> float:
    """
    Fetch headlines for a ticker and return average sentiment score.
    Falls back to 0 (neutral) if NewsAPI not configured.
    """
    if not news_api_key:
        return 0.0
    try:
        from newsapi import NewsApiClient
        base  = ticker.split(".")[0]
        client = NewsApiClient(api_key=news_api_key)
        from datetime import datetime, timedelta
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        resp = client.get_everything(q=base, language="en", sort_by="publishedAt",
                                     page_size=10, from_param=from_date)
        articles = resp.get("articles", [])
        if not articles:
            return 0.0
        scores = [score_sentiment(a.get("title", ""))["score"] for a in articles]
        return float(np.mean(scores))
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════
#  FULL STOCK DEEP ANALYSIS
# ════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _cached_fetch(ticker: str, start: str, end: str):
    """Cache 10-year data fetches so repeated calls within session are instant."""
    df = fetch_stock_data(ticker, start, end)
    return df

def deep_analyse_stock(ticker: str, news_api_key: str = "") -> dict:
    """
    Run a complete 10-year deep analysis on a single ticker.
    Returns a rich dict with all metrics, patterns, and composite score.
    """
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    df = fetch_stock_data(ticker, start, end)
    if df is None or df.empty or len(df) < 60:
        return {"ticker": ticker, "error": "Insufficient data"}

    df = add_all_indicators(df)
    df.dropna(inplace=True)

    if len(df) < 30:
        return {"ticker": ticker, "error": "Too few rows after indicator calculation"}

    close = df["Close"]
    last  = df.iloc[-1]

    def ret(n):
        return round(((close.iloc[-1] - close.iloc[-n-1]) / close.iloc[-n-1]) * 100, 2) if len(close) > n else None

    # Annual returns (up to 10 years)
    annual_returns = []
    for yr in range(1, 11):
        idx = min(252 * yr, len(close) - 1)
        try:
            r = round(((float(close.iloc[-1]) - float(close.iloc[-idx-1])) / float(close.iloc[-idx-1])) * 100, 2)
            annual_returns.append(r)
        except Exception:
            break

    # Yearly breakdown
    df["_yr"] = df.index.year
    yearly = df.groupby("_yr")["Close"].apply(
        lambda x: round(((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100, 2) if len(x) > 1 else 0
    )
    yearly_dict = yearly.tail(10).to_dict()

    # Volatility
    daily_ret  = close.pct_change().dropna()
    volatility = round(daily_ret.std() * np.sqrt(252) * 100, 2)

    # Max Drawdown
    rolling_max  = close.cummax()
    drawdown     = ((close - rolling_max) / rolling_max) * 100
    max_drawdown = round(drawdown.min(), 2)

    # Support / Resistance
    w52_high = round(float(df["High"].iloc[-252:].max()), 2) if len(df) >= 252 else None
    w52_low  = round(float(df["Low"].iloc[-252:].min()),  2) if len(df) >= 252 else None

    # Average True Range (risk measure)
    atr = round(float(last["ATR"]), 2) if "ATR" in df.columns else None

    # News sentiment
    news_score = get_news_sentiment_score(ticker, news_api_key)

    # Patterns
    patterns = detect_patterns(df)

    # Composite score
    composite = compute_composite_score(df, patterns, news_score)

    # Win rate: % of years with positive returns
    win_rate = round(sum(1 for v in yearly_dict.values() if v > 0) / max(len(yearly_dict), 1) * 100, 0)

    return {
        "ticker":          ticker,
        "last_close":      round(float(close.iloc[-1]), 2),
        "data_years":      round(len(df) / 252, 1),
        # Momentum
        "ret_1d":          ret(1),
        "ret_5d":          ret(5),
        "ret_20d":         ret(20),
        "ret_60d":         ret(60),
        "ret_1y":          ret(252),
        # Annual history
        "annual_returns":  annual_returns,
        "yearly_breakdown":yearly_dict,
        "win_rate_pct":    win_rate,
        # Risk
        "volatility":      volatility,
        "max_drawdown":    max_drawdown,
        "atr":             atr,
        # Technicals
        "rsi":             round(float(last["RSI"]),         2) if "RSI"         in df.columns else None,
        "macd":            round(float(last["MACD"]),        4) if "MACD"        in df.columns else None,
        "macd_hist":       round(float(last["MACD_Hist"]),   4) if "MACD_Hist"   in df.columns else None,
        "ema20":           round(float(last["EMA_20"]),      2) if "EMA_20"      in df.columns else None,
        "ema50":           round(float(last["EMA_50"]),      2) if "EMA_50"      in df.columns else None,
        "bb_pct":          round(float(last["BB_Pct"]),      3) if "BB_Pct"      in df.columns else None,
        "bb_width":        round(float(last["BB_Width"]),    4) if "BB_Width"    in df.columns else None,
        "stoch_k":         round(float(last["Stoch_K"]),     1) if "Stoch_K"     in df.columns else None,
        "stoch_d":         round(float(last["Stoch_D"]),     1) if "Stoch_D"     in df.columns else None,
        # 52-week
        "w52_high":        w52_high,
        "w52_low":         w52_low,
        "pct_from_52w_high": round(((float(close.iloc[-1]) - w52_high) / w52_high) * 100, 2) if w52_high else None,
        # News
        "news_sentiment_score": round(news_score, 2),
        # Patterns
        "patterns":        patterns,
        # Composite prediction
        "composite_score": composite["total"],
        "outlook":         composite["outlook"],
        "confidence":      composite["confidence"],
        "score_breakdown": composite["breakdown"],
    }


# ════════════════════════════════════════════════════════════
#  PROMPT BUILDER  (token-budget aware)
# ════════════════════════════════════════════════════════════

# Gemini free tier limit: ~30k tokens input. We target <4000 to be safe.
_TOKEN_BUDGET = 3500

SYSTEM_PROMPT = (
    "You are StockIQ, an expert Indian stock market analyst. "
    "You receive pre-computed analysis data (composite scores, technicals, patterns, history) "
    "for NSE stocks and answer questions with specific, data-driven predictions. "
    "Always structure responses with: ## Summary | ## Stock Rankings | ## Verdict | ## Risks. "
    "Reference exact numbers. End with the disclaimer: "
    "'⚠️ AI analysis for educational purposes only — not financial advice.'"
)


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def format_stock_for_prompt(s: dict) -> str:
    """Ultra-compact single-line format — ~60 tokens per stock."""
    if "error" in s:
        return f"[{s['ticker']}] ERR:{s['error'][:40]}"

    pats_raw = s.get("patterns", [])
    # Only keep bullish/bearish pattern keywords, drop decorative text
    pats = "; ".join(
        p.split("—")[0].strip().replace("🟢","▲").replace("🔴","▼").replace("⚡","~")
        for p in pats_raw[:3]
    ) or "none"

    yb = s.get("yearly_breakdown", {})
    yb_str = ",".join(f"{yr}:{v:+.0f}%" for yr, v in sorted(yb.items())[-3:])

    return (
        f"{s['ticker']}|Score:{s['composite_score']}|{s['outlook']}|Conf:{s['confidence']}|"
        f"P:₹{s['last_close']}|WR:{s['win_rate_pct']}%|"
        f"1D:{s.get('ret_1d',0):+.1f}%|5D:{s.get('ret_5d',0):+.1f}%|"
        f"20D:{s.get('ret_20d',0):+.1f}%|1Y:{s.get('ret_1y',0):+.1f}%|"
        f"RSI:{s.get('rsi','?')}|MACD:{s.get('macd_hist','?')}|"
        f"StochK:{s.get('stoch_k','?')}|BB%:{s.get('bb_pct','?')}|"
        f"Vol:{s.get('volatility','?')}%|DD:{s.get('max_drawdown','?')}%|"
        f"52WH%:{s.get('pct_from_52w_high','?')}%|News:{s.get('news_sentiment_score',0):+.1f}|"
        f"Yrs:[{yb_str}]|Pat:[{pats[:80]}]"
    )


def build_prompt(valid: list, errors: list, active_ticker: str, user_question: str) -> str:
    """Build a prompt that stays within _TOKEN_BUDGET tokens."""
    header = (
        f"Date:{date.today().strftime('%d-%b-%Y')} "
        f"ActiveTicker:{active_ticker} "
        f"Stocks:{len(valid)}ok/{len(errors)}err\n"
        "FORMAT: one line per stock, pipe-separated fields.\n"
    )

    # Add stocks one by one until budget is close to full
    lines = []
    budget_used = _approx_tokens(SYSTEM_PROMPT) + _approx_tokens(header) + _approx_tokens(user_question) + 200
    for s in valid:
        line = format_stock_for_prompt(s)
        cost = _approx_tokens(line)
        if budget_used + cost > _TOKEN_BUDGET:
            lines.append(f"[truncated {len(valid)-len(lines)} more stocks to fit token limit]")
            break
        lines.append(line)
        budget_used += cost

    for e in errors:
        lines.append(format_stock_for_prompt(e))

    stock_block = "\n".join(lines)
    return (
        f"{header}"
        f"{stock_block}\n\n"
        f"QUESTION: {user_question}\n"
        f"Answer using the data above. Be specific with numbers."
    )


# ════════════════════════════════════════════════════════════
#  MAIN ASK FUNCTION
# ════════════════════════════════════════════════════════════

def ask_ai_analyst(
    user_question: str,
    active_ticker: str,
    extra_tickers: list = None,
    news_api_key: str = "",
    gemini_api_key: str = "",
) -> str:
    """
    Deep-analyse stocks and get AI prediction via Google Gemini API.
    All features preserved: parallel fetch, LRU cache, pattern detection,
    composite scoring, news sentiment, 10-year history.
    """
    import time as _time

    global GEMINI_API_KEY
    if gemini_api_key:
        GEMINI_API_KEY = gemini_api_key

    # ── Determine tickers to scan ─────────────────────────────
    tickers_to_scan = [active_ticker]
    if extra_tickers:
        tickers_to_scan += [t for t in extra_tickers if t not in tickers_to_scan]

    broad_keywords = [
        "which stock", "best stock", "top stock", "recommend", "5 stock",
        "gain", "profit next", "rise next", "grow next", "invest",
        "pick", "gainer", "loser", "winner", "buy today",
        "next few days", "this week", "short term",
    ]
    if any(kw in user_question.lower() for kw in broad_keywords) and not extra_tickers:
        tickers_to_scan = WATCHLIST[:6]   # keep small for token budget

    # ── Parallel deep analysis ────────────────────────────────
    def _analyse_safe(sym):
        try:
            return deep_analyse_stock(sym, news_api_key)
        except Exception as e:
            return {"ticker": sym, "error": str(e)}

    max_workers = min(6, len(tickers_to_scan))
    analyses = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyse_safe, sym): sym for sym in tickers_to_scan}
        for future in as_completed(futures):
            analyses.append(future.result())

    valid  = sorted([a for a in analyses if "error" not in a],
                    key=lambda x: x.get("composite_score", 0), reverse=True)
    errors = [a for a in analyses if "error" in a]

    # ── Build prompt (token-budget aware) ────────────────────
    full_prompt    = build_prompt(valid, errors, active_ticker, user_question)
    combined_prompt = SYSTEM_PROMPT + "\n\n" + full_prompt

    # Safety check — if still too big, trim to top 3 stocks only
    if _approx_tokens(combined_prompt) > _TOKEN_BUDGET:
        full_prompt    = build_prompt(valid[:3], [], active_ticker, user_question)
        combined_prompt = SYSTEM_PROMPT + "\n\n" + full_prompt

    # ── Call Gemini API with retry/backoff ────────────────────
    api_key = GEMINI_API_KEY
    if not api_key:
        return (
            "❌ **Gemini API key not found.**\n\n"
            "Add `GEMINI_API_KEY=AIza...` to your `.env` file and restart Streamlit."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": combined_prompt}]}],
        "generationConfig": {
            "temperature":     0.35,
            "maxOutputTokens": 1200,
            "topP":            0.9,
        },
    }

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )

            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)   # 4s → 8s → 16s
                _time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    return (
                        "⚠️ **Gemini rate limit hit (429).**\n\n"
                        "Free tier allows ~15 req/min. Wait 30s and retry.\n"
                        "Upgrade at: https://aistudio.google.com"
                    )
                continue

            if resp.status_code == 400:
                # Log the actual error for debugging, then retry with minimal prompt
                err_body = resp.text[:300]
                if attempt < MAX_RETRIES - 1:
                    # Shrink to single stock and retry
                    mini = build_prompt(valid[:1], [], active_ticker, user_question)
                    payload["contents"] = [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + mini}]}]
                    _time.sleep(1)
                    continue
                return (
                    f"❌ **Gemini API error (400).**\n"
                    f"Detail: {err_body}\n\n"
                    "Try asking about a single stock, e.g. 'Analyse RELIANCE.NS'."
                )

            if resp.status_code == 403:
                return (
                    "❌ **Invalid Gemini API key (403).**\n"
                    "Check `GEMINI_API_KEY` in your `.env` file."
                )

            resp.raise_for_status()
            data = resp.json()

            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return f"❌ Unexpected Gemini response format:\n{str(data)[:400]}"

        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                return "⏱️ **Request timed out.** Try asking about a single stock."
            _time.sleep(3)
        except Exception as e:
            return f"❌ **AI Analyst error:** {str(e)}"

    return "❌ Failed after retries. Please try again in a moment."