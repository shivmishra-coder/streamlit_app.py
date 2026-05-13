# ============================================================
# data_engine.py
# Handles all market data fetching (via yfinance) and
# technical indicator calculations (EMA, RSI, MACD).
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Data Fetching ────────────────────────────────────────────

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df.empty:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().title() for c in df.columns]

        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            return pd.DataFrame()

        df = df[[c for c in df.columns if c in required]]
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"[data_engine] fetch_stock_data error: {e}")
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or len(info) < 5:
            try:
                fi = t.fast_info
                info = {
                    "longName": ticker, "sector": "N/A",
                    "currency": getattr(fi, "currency", "₹"),
                    "marketCap": getattr(fi, "market_cap", None),
                }
            except Exception:
                info = {"longName": ticker, "sector": "N/A", "currency": "₹"}
        return info
    except Exception as e:
        print(f"[data_engine] fetch_stock_info error: {e}")
        return {}


# ── Technical Indicators ─────────────────────────────────────

def add_ema(df: pd.DataFrame, spans: list = [20, 50]) -> pd.DataFrame:
    for span in spans:
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast        = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow        = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]      = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["BB_Mid"]   = df["Close"].rolling(period).mean()
    df["BB_Std"]   = df["Close"].rolling(period).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["Close"].diff())
    df["OBV"]  = (direction * df["Volume"]).cumsum()
    return df


# ── Volume Profile & Smart Money ─────────────────────────────

def add_volume_profile(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Volume SMA and Volume Surge indicator.
    vol_surge_ratio > 1.5 = institutional / smart-money activity.
    Differentiates 'Price+Volume' (conviction move) vs 'Low Volume Rally' (weak).
    """
    df["Vol_SMA20"]     = df["Volume"].rolling(window).mean()
    df["Vol_Surge"]     = df["Volume"] / df["Vol_SMA20"].replace(0, np.nan)
    # Price action quality label
    price_up = df["Close"] > df["Close"].shift(1)
    vol_high = df["Vol_Surge"] >= 1.5
    df["Vol_Quality"] = np.where(
        price_up &  vol_high, "Confirmed Rally",
        np.where(price_up & ~vol_high, "Low Volume Rally",
        np.where(~price_up & vol_high, "High Vol Decline",
        "Weak Decline"))
    )
    return df


# ── Dynamic Risk-Reward (ATR-based) ──────────────────────────

def compute_risk_reward(df: pd.DataFrame, atr_mult: float = 1.5,
                        rr_target: float = 2.0) -> dict:
    """
    ATR-based Stop-Loss and Risk-to-Reward calculator.

    Args:
        df:         Enriched OHLCV DataFrame (must contain ATR column).
        atr_mult:   Multiplier for stop-loss distance below entry (default 1.5x ATR).
        rr_target:  Minimum R:R ratio to flag as 'Strong Buy' (default 2:1).

    Returns:
        dict with keys:
            entry_price, stop_loss, risk_per_share,
            target_1r, target_2r, target_3r,
            rr_ratio, signal_quality ('Strong Buy' | 'Buy' | 'Neutral')
            atr_value
    """
    if df.empty or "ATR" not in df.columns:
        return {"error": "ATR not available"}

    last         = df.iloc[-1]
    entry        = float(last["Close"])
    atr_val      = float(last["ATR"])
    stop_loss    = round(entry - atr_mult * atr_val, 2)
    risk         = round(entry - stop_loss, 2)

    # Upside targets using projected ML/momentum move (use 5-day return as proxy)
    if len(df) >= 6:
        avg_5d_move = abs(float(df["Close"].pct_change(5).iloc[-1]) * entry)
    else:
        avg_5d_move = atr_val * 2

    target_1r = round(entry + risk,            2)   # 1:1
    target_2r = round(entry + risk * 2,        2)   # 2:1
    target_3r = round(entry + risk * 3,        2)   # 3:1

    projected_upside = avg_5d_move
    rr_ratio = round(projected_upside / risk, 2) if risk > 0 else 0.0

    if rr_ratio >= rr_target:
        signal_quality = "Strong Buy"
    elif rr_ratio >= 1.0:
        signal_quality = "Buy"
    else:
        signal_quality = "Neutral"

    return {
        "entry_price":    round(entry, 2),
        "stop_loss":      stop_loss,
        "risk_per_share": risk,
        "target_1r":      target_1r,
        "target_2r":      target_2r,
        "target_3r":      target_3r,
        "rr_ratio":       rr_ratio,
        "signal_quality": signal_quality,
        "atr_value":      round(atr_val, 2),
    }


# ── Market Regime & Sector Correlation ───────────────────────

# Sector index mapping for NSE tickers
SECTOR_INDEX_MAP = {
    "HDFCBANK.NS": "^NSEBANK", "ICICIBANK.NS": "^NSEBANK", "SBIN.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK", "BAJFINANCE.NS": "^NSEBANK",
    "TCS.NS":  "^CNXIT", "INFY.NS": "^CNXIT", "WIPRO.NS": "^CNXIT",
    "HCLTECH.NS": "^CNXIT", "TECHM.NS": "^CNXIT", "LTIM.NS": "^CNXIT",
    "SUNPHARMA.NS": "^CNXPHARMA",
    "MARUTI.NS": "^CNXAUTO", "TATAMOTORS.NS": "^CNXAUTO", "M&M.NS": "^CNXAUTO",
    "RELIANCE.NS": "^CNX500", "ADANIENT.NS": "^CNX500",
    "COALINDIA.NS": "^CNXENERGY", "ONGC.NS": "^CNXENERGY", "NTPC.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "ITC.NS": "^CNXFMCG", "HINDUNILVR.NS": "^CNXFMCG", "NESTLEIND.NS": "^CNXFMCG",
    "BHARTIARTL.NS": "^CNX500",
}

_NIFTY50 = "^NSEI"


def fetch_market_regime(lookback_days: int = 220) -> dict:
    """
    Fetch Nifty 50 data and determine market regime.
    Regime = 'Bull' if Nifty > 200-day EMA, else 'Bear'.
    Returns a cautionary multiplier (0.7) in Bear regime.
    """
    try:
        from datetime import date, timedelta
        end   = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df    = fetch_stock_data(_NIFTY50, start, end)
        if df.empty or len(df) < 40:
            return {"regime": "Unknown", "multiplier": 1.0, "nifty_price": None,
                    "nifty_ema200": None, "pct_from_ema200": None}

        ema200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        price  = float(df["Close"].iloc[-1])
        pct    = round((price - ema200) / ema200 * 100, 2)

        if price > ema200:
            regime, mult = "Bull", 1.0
        else:
            regime, mult = "Bear", 0.7   # cautionary multiplier on buy signals

        return {
            "regime":          regime,
            "multiplier":      mult,
            "nifty_price":     round(price, 2),
            "nifty_ema200":    round(ema200, 2),
            "pct_from_ema200": pct,
        }
    except Exception as e:
        print(f"[data_engine] fetch_market_regime error: {e}")
        return {"regime": "Unknown", "multiplier": 1.0, "nifty_price": None,
                "nifty_ema200": None, "pct_from_ema200": None}


def fetch_sector_performance(ticker: str, lookback_days: int = 10) -> dict:
    """
    Compare ticker's 5-day performance vs its sectoral index.
    Returns outperformance delta and a sector_boost score.
    """
    try:
        from datetime import date, timedelta
        sector_sym = SECTOR_INDEX_MAP.get(ticker.upper())
        if not sector_sym:
            return {"sector_index": None, "stock_ret5": 0.0,
                    "sector_ret5": 0.0, "outperformance": 0.0, "sector_boost": 0}

        end   = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")

        df_sec = fetch_stock_data(sector_sym, start, end)
        df_stk = fetch_stock_data(ticker, start, end)

        def ret5(df):
            if df.empty or len(df) < 5:
                return 0.0
            return round(((float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-6]))
                          / float(df["Close"].iloc[-6])) * 100, 2)

        stock_ret  = ret5(df_stk)
        sector_ret = ret5(df_sec)
        out        = round(stock_ret - sector_ret, 2)

        # Boost score: +2 if outperforming by >1%, +1 if even, 0 if underperforming
        boost = 2 if out > 1 else (1 if out >= 0 else 0)

        return {
            "sector_index":    sector_sym,
            "stock_ret5":      stock_ret,
            "sector_ret5":     sector_ret,
            "outperformance":  out,
            "sector_boost":    boost,
        }
    except Exception as e:
        print(f"[data_engine] fetch_sector_performance error: {e}")
        return {"sector_index": None, "stock_ret5": 0.0,
                "sector_ret5": 0.0, "outperformance": 0.0, "sector_boost": 0}


# ── Weighted Composite Scoring Engine ────────────────────────

def compute_weighted_composite(
    technical_score: float,   # raw 0–100
    historical_win_rate: float,  # 0–100
    news_score: float,        # raw sentiment (-5 to +5 typical)
    volume_surge_ratio: float,   # 1.0 = normal; 1.5+ = surge
    sector_boost: int,           # 0, 1, or 2
    market_multiplier: float = 1.0,  # 1.0 = bull; 0.7 = bear regime
) -> dict:
    """
    5-component weighted composite scoring system:
      Technical Indicators : 30%
      Historical Win-Rate  : 20%
      News Sentiment       : 20%
      Volume Strength      : 15%
      Sector/Market Align  : 15%

    Returns dict with weighted_score (0–100), conviction_label,
    component breakdown, and reasons list.
    """
    reasons = []

    # ── 1. Technical (30 pts) ─────────────────────────────────
    tech_norm  = float(np.clip(technical_score, 0, 100))
    tech_pts   = tech_norm * 0.30
    if tech_norm >= 70:
        reasons.append("Strong technical setup")
    elif tech_norm >= 50:
        reasons.append("Moderate technical signals")

    # ── 2. Historical Win-Rate (20 pts) ──────────────────────
    hist_pts = float(np.clip(historical_win_rate, 0, 100)) * 0.20
    if historical_win_rate >= 70:
        reasons.append(f"Strong 10Y historical win-rate ({historical_win_rate:.0f}%)")
    elif historical_win_rate >= 55:
        reasons.append(f"Decent 10Y win-rate ({historical_win_rate:.0f}%)")

    # ── 3. News Sentiment (20 pts) ────────────────────────────
    # news_score typically -5 to +5; normalise to 0–100
    news_norm  = float(np.clip((news_score + 5) / 10 * 100, 0, 100))
    news_pts   = news_norm * 0.20
    if news_score > 1:
        reasons.append("Positive news sentiment")
    elif news_score < -1:
        reasons.append("Negative news headwinds")

    # ── 4. Volume Strength (15 pts) ──────────────────────────
    vol_norm = float(np.clip((volume_surge_ratio - 0.5) / 2.0 * 100, 0, 100))
    vol_pts  = vol_norm * 0.15
    if volume_surge_ratio >= 1.5:
        reasons.append(f"Volume breakout ({volume_surge_ratio:.1f}x avg) — smart money signal")
    elif volume_surge_ratio >= 1.2:
        reasons.append(f"Elevated volume ({volume_surge_ratio:.1f}x avg)")

    # ── 5. Sector / Market Alignment (15 pts) ────────────────
    # sector_boost 0/1/2 → 0–100 scale, then apply market regime multiplier
    sector_norm = float(np.clip(sector_boost / 2 * 100, 0, 100))
    sector_pts  = sector_norm * 0.15 * market_multiplier
    if market_multiplier < 1.0:
        reasons.append("⚠️ Nifty below 200-EMA — cautionary regime (signals discounted)")
    if sector_boost >= 2:
        reasons.append("Sector tailwinds — stock outperforming its index")
    elif sector_boost == 1:
        reasons.append("In-line with sector performance")

    total = round(tech_pts + hist_pts + news_pts + vol_pts + sector_pts, 1)

    if total >= 70:
        label = "Very High Conviction"
    elif total >= 55:
        label = "High Conviction"
    elif total >= 40:
        label = "Moderate Conviction"
    elif total >= 25:
        label = "Low Conviction"
    else:
        label = "Very Low Conviction"

    return {
        "weighted_score":   total,
        "conviction_label": label,
        "reasons":          reasons,
        "breakdown": {
            "technical":   round(tech_pts,    1),
            "historical":  round(hist_pts,    1),
            "news":        round(news_pts,    1),
            "volume":      round(vol_pts,     1),
            "sector":      round(sector_pts,  1),
        },
    }


def enrich_data(df: pd.DataFrame,
                show_ema=True, show_rsi=True, show_macd=True) -> pd.DataFrame:
    if show_ema:
        df = add_ema(df, spans=[20, 50])
    if show_rsi:
        df = add_rsi(df)
    if show_macd:
        df = add_macd(df)
    return df


def enrich_full(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ALL indicators needed for deep screening."""
    df = add_ema(df, spans=[20, 50, 200])
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_volume_profile(df)
    return df


# ── Watchlist ────────────────────────────────────────────────

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "TATAMOTORS.NS", "WIPRO.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "LTIM.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
    "ADANIENT.NS", "COALINDIA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "HCLTECH.NS", "TECHM.NS", "BHARTIARTL.NS", "ITC.NS", "M&M.NS",
]


# ════════════════════════════════════════════════════════════
#  DEEP SCREENER  (10-year data, all indicators, ML signal)
# ════════════════════════════════════════════════════════════

def _ml_signal(df: pd.DataFrame) -> float:
    """
    Quick Random-Forest signal: returns predicted % change for next day.
    Positive → bullish ML signal, negative → bearish.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        feat = df.copy()
        for lag in range(1, 6):
            feat[f"Lag_{lag}"] = feat["Close"].shift(lag)
        feat["Roll5"]  = feat["Close"].rolling(5).mean()
        feat["Roll20"] = feat["Close"].rolling(20).mean()
        feat["Ret"]    = feat["Close"].pct_change()
        feat["Target"] = feat["Close"].shift(-1)
        feat.dropna(inplace=True)

        cols = [c for c in feat.columns if c not in ("Target", "Open", "High", "Low", "Volume")]
        X, y = feat[cols].values, feat["Target"].values
        if len(X) < 80:
            return 0.0

        split = int(len(X) * 0.8)
        mdl = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        mdl.fit(X[:split], y[:split])

        pred  = float(mdl.predict(X[-1].reshape(1, -1))[0])
        last  = float(feat["Close"].iloc[-1])
        return round(((pred - last) / last) * 100, 2)
    except Exception:
        return 0.0


def _news_sentiment_score(ticker: str, news_api_key: str = "") -> float:
    """Return average sentiment score from recent headlines (-1 to +1 scaled)."""
    if not news_api_key:
        return 0.0
    try:
        from newsapi import NewsApiClient
        from news_service import score_sentiment
        from datetime import datetime, timedelta
        base   = ticker.split(".")[0]
        client = NewsApiClient(api_key=news_api_key)
        from_d = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        resp   = client.get_everything(q=base, language="en",
                                       sort_by="publishedAt",
                                       page_size=10, from_param=from_d)
        arts = resp.get("articles", [])
        if not arts:
            return 0.0
        scores = [score_sentiment(a.get("title", ""))["score"] for a in arts]
        return float(np.mean(scores))
    except Exception:
        return 0.0


def _analyse_one(sym: str, news_api_key: str = "") -> dict | None:
    """
    Full 10-year deep analysis for a single ticker.
    Returns a rich dict or None on failure.
    """
    from datetime import date, timedelta
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    df = fetch_stock_data(sym, start, end)
    if df is None or df.empty or len(df) < 120:
        return None

    df = enrich_full(df)
    df.dropna(inplace=True)
    if len(df) < 60:
        return None

    close = df["Close"]
    last  = df.iloc[-1]

    # ── 10-year yearly returns ────────────────────────────────
    df["_yr"] = df.index.year
    yearly = df.groupby("_yr")["Close"].apply(
        lambda x: round(((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100, 2)
        if len(x) > 1 else 0
    ).tail(10)
    win_rate    = round(sum(1 for v in yearly if v > 0) / max(len(yearly), 1) * 100, 0)
    avg_yr_ret  = round(float(yearly.mean()), 2)
    best_yr     = round(float(yearly.max()), 2)
    worst_yr    = round(float(yearly.min()), 2)

    # ── Momentum ─────────────────────────────────────────────
    def ret(n):
        return round(((float(close.iloc[-1]) - float(close.iloc[-n-1]))
                      / float(close.iloc[-n-1])) * 100, 2) if len(close) > n else 0.0

    r1, r5, r20, r60, r252 = ret(1), ret(5), ret(20), ret(60), ret(252)

    # ── Volatility & Drawdown ────────────────────────────────
    daily_ret  = close.pct_change().dropna()
    volatility = round(float(daily_ret.std() * np.sqrt(252) * 100), 2)
    roll_max   = close.cummax()
    max_dd     = round(float(((close - roll_max) / roll_max).min() * 100), 2)

    # ── Latest indicator values ───────────────────────────────
    rsi       = round(float(last["RSI"]),         2) if "RSI"      in df.columns else 50.0
    macd_hist = round(float(last["MACD_Hist"]),   4) if "MACD_Hist" in df.columns else 0.0
    ema20     = round(float(last["EMA_20"]),       2) if "EMA_20"   in df.columns else float(close.iloc[-1])
    ema50     = round(float(last["EMA_50"]),       2) if "EMA_50"   in df.columns else float(close.iloc[-1])
    ema200    = round(float(last.get("EMA_200", close.iloc[-1])), 2)
    bb_pct    = round(float(last["BB_Pct"]),       3) if "BB_Pct"   in df.columns else 0.5
    bb_width  = round(float(last["BB_Width"]),     4) if "BB_Width" in df.columns else 0.0
    px        = round(float(close.iloc[-1]), 2)

    # ── OBV trend ────────────────────────────────────────────
    obv_trend = 0.0
    if "OBV" in df.columns and len(df) >= 10:
        obv_trend = float(df["OBV"].iloc[-5:].mean() - df["OBV"].iloc[-10:-5].mean())

    # ── Pattern signals ──────────────────────────────────────
    patterns  = []
    if "EMA_20" in df.columns and "EMA_50" in df.columns and len(df) >= 2:
        prev = df.iloc[-2]
        if prev["EMA_20"] < prev["EMA_50"] and last["EMA_20"] > last["EMA_50"]:
            patterns.append("Golden Cross")
        if prev["EMA_20"] > prev["EMA_50"] and last["EMA_20"] < last["EMA_50"]:
            patterns.append("Death Cross")
    if "BB_Width" in df.columns:
        if bb_pct > 0.95: patterns.append("BB Upper (overbought)")
        if bb_pct < 0.05: patterns.append("BB Lower (oversold)")
        if bb_width < df["BB_Width"].iloc[-20:].quantile(0.15):
            patterns.append("BB Squeeze")
    if "MACD" in df.columns and len(df) >= 2:
        prev = df.iloc[-2]
        if prev["MACD"] < prev["MACD_Signal"] and last["MACD"] > last["MACD_Signal"]:
            patterns.append("MACD Bull Cross")
        if prev["MACD"] > prev["MACD_Signal"] and last["MACD"] < last["MACD_Signal"]:
            patterns.append("MACD Bear Cross")

    # ── Volume surge ─────────────────────────────────────────
    avg_vol = float(df["Volume"].iloc[-20:].mean())
    vol_ratio = round(float(last["Volume"]) / avg_vol, 2) if avg_vol > 0 else 1.0
    vol_quality = str(last.get("Vol_Quality", "N/A")) if "Vol_Quality" in df.columns else "N/A"

    # ── ML signal ────────────────────────────────────────────
    ml_pct = _ml_signal(df)

    # ── News sentiment ────────────────────────────────────────
    news_score = _news_sentiment_score(sym, news_api_key)

    # ── ATR-based Risk-Reward ─────────────────────────────────
    rr_data = compute_risk_reward(df)

    # ── Market Regime (Nifty 50 filter) ──────────────────────
    # NOTE: fetched once in deep_screen_stocks and passed in; fallback here
    market_regime = fetch_market_regime()
    mkt_multiplier = market_regime.get("multiplier", 1.0)

    # ── Sector correlation ────────────────────────────────────
    sector_data  = fetch_sector_performance(sym)
    sector_boost = sector_data.get("sector_boost", 0)

    # ════════════════════════════════════════════════════════
    #  COMPOSITE BULLISH / BEARISH SCORE  (-100 to +100)
    # ════════════════════════════════════════════════════════
    score = 0.0

    # Momentum (30 pts)
    score += np.clip(r1  * 2,    -6,  6)
    score += np.clip(r5  * 1.5,  -7,  7)
    score += np.clip(r20 * 0.8,  -8,  8)
    score += np.clip(r60 * 0.3,  -9,  9)

    # Trend (25 pts)
    if px > ema20:  score += 8
    else:           score -= 8
    if px > ema50:  score += 7
    else:           score -= 7
    if ema20 > ema50: score += 10
    else:             score -= 10

    # Oscillators (20 pts)
    if rsi < 30:    score += 10
    elif rsi < 45:  score += 5
    elif rsi > 70:  score -= 10
    elif rsi > 55:  score -= 3
    score += np.clip(macd_hist * 5, -10, 10)

    # Patterns (10 pts)
    for p in patterns:
        if "Bull" in p or "Golden" in p or "Squeeze" in p or "oversold" in p:
            score += 3
        elif "Bear" in p or "Death" in p or "overbought" in p:
            score -= 3

    # ML signal (10 pts)
    score += np.clip(ml_pct * 2, -10, 10)

    # News (5 pts)
    score += np.clip(news_score * 2.5, -5, 5)

    # 10-year history factor (extra confidence adjuster)
    score += np.clip(avg_yr_ret * 0.05, -5, 5)  # long-run compounder bonus/penalty

    # ── Volume Surge Bonus (smart money detection) ────────────
    if vol_ratio >= 1.5:
        if r5 > 0:   # price + volume = confirmed rally
            score += 5   # conviction boost
        else:         # high volume decline = bearish pressure
            score -= 3

    # ── Sector outperformance bonus ───────────────────────────
    score += sector_boost * 2   # up to +4 pts

    # ── Market regime: apply cautionary multiplier to bullish signals ──
    if mkt_multiplier < 1.0 and score > 0:
        score = score * mkt_multiplier   # discount buy signals in bear market

    total = round(float(np.clip(score, -100, 100)), 1)

    # ── NEW: 5-component weighted composite (0–100 scale) ─────
    # Convert -100/+100 score to 0-100 for technical component
    tech_0_100 = float(np.clip((score + 100) / 2, 0, 100))
    weighted   = compute_weighted_composite(
        technical_score    = tech_0_100,
        historical_win_rate= win_rate,
        news_score         = news_score,
        volume_surge_ratio = vol_ratio,
        sector_boost       = sector_boost,
        market_multiplier  = mkt_multiplier,
    )

    # ── Bearish-specific scoring (for screener) ───────────────
    bear_score = 0
    bear_reasons = []

    if r5 < -1:
        bear_reasons.append(f"5D ret: {r5:+.1f}%"); bear_score += 2
    elif r5 < 0:
        bear_reasons.append(f"5D ret: {r5:+.1f}%"); bear_score += 1
    if rsi > 65:
        bear_reasons.append(f"RSI overbought ({rsi:.0f})"); bear_score += 1
    if rsi > 70:
        bear_score += 1  # extra
    if macd_hist < 0:
        bear_reasons.append("MACD hist negative"); bear_score += 1
    if "MACD Bear Cross" in patterns:
        bear_reasons.append("MACD bearish crossover"); bear_score += 2
    if "Death Cross" in patterns:
        bear_reasons.append("Death Cross (EMA)"); bear_score += 2
    if px < ema20:
        bear_reasons.append("Price below EMA20"); bear_score += 1
    if px < ema50:
        bear_reasons.append("Price below EMA50"); bear_score += 1
    if news_score < -0.5:
        bear_reasons.append(f"Negative news sentiment ({news_score:.1f})"); bear_score += 1
    if ml_pct < -0.5:
        bear_reasons.append(f"ML bearish signal ({ml_pct:+.1f}%)"); bear_score += 1
    if worst_yr < -25:
        bear_reasons.append(f"History: worst yr {worst_yr:.0f}%"); bear_score += 1

    # ── Bullish-specific scoring ──────────────────────────────
    bull_score = 0
    bull_reasons = []

    if r5 > 1:
        bull_reasons.append(f"5D ret: {r5:+.1f}%"); bull_score += 2
    elif r5 > 0:
        bull_reasons.append(f"5D ret: {r5:+.1f}%"); bull_score += 1
    if rsi < 35:
        bull_reasons.append(f"RSI oversold ({rsi:.0f})"); bull_score += 2
    elif rsi < 50:
        bull_reasons.append(f"RSI neutral-low ({rsi:.0f})"); bull_score += 1
    if macd_hist > 0:
        bull_reasons.append("MACD hist positive"); bull_score += 1
    if "MACD Bull Cross" in patterns:
        bull_reasons.append("MACD bullish crossover"); bull_score += 2
    if "Golden Cross" in patterns:
        bull_reasons.append("Golden Cross (EMA)"); bull_score += 2
    if "BB Squeeze" in patterns:
        bull_reasons.append("Bollinger Squeeze (breakout pending)"); bull_score += 1
    if px > ema20 and px > ema50:
        bull_reasons.append("Price above EMA20 & EMA50"); bull_score += 2
    if news_score > 0.5:
        bull_reasons.append(f"Positive news sentiment ({news_score:.1f})"); bull_score += 1
    if ml_pct > 0.3:
        bull_reasons.append(f"ML bullish signal ({ml_pct:+.1f}%)"); bull_score += 1
    if win_rate >= 60:
        bull_reasons.append(f"10Y win rate {win_rate:.0f}%"); bull_score += 1
    if avg_yr_ret > 10:
        bull_reasons.append(f"10Y avg annual return {avg_yr_ret:.1f}%"); bull_score += 1
    if obv_trend > 0 and r5 > 0:
        bull_reasons.append("OBV confirms uptrend"); bull_score += 1
    if vol_ratio >= 1.5 and r5 > 0:
        bull_reasons.append(f"Volume surge ({vol_ratio:.1f}x avg) — confirmed rally"); bull_score += 2
    elif vol_ratio >= 1.2 and r5 > 0:
        bull_reasons.append(f"Elevated volume ({vol_ratio:.1f}x avg)"); bull_score += 1
    if sector_boost >= 2:
        bull_reasons.append("Outperforming sectoral index"); bull_score += 1
    if not isinstance(rr_data, dict) or "error" not in rr_data:
        if rr_data.get("signal_quality") == "Strong Buy":
            bull_reasons.append(f"R:R={rr_data['rr_ratio']:.1f} — meets Strong Buy threshold"); bull_score += 2
        elif rr_data.get("signal_quality") == "Buy":
            bull_reasons.append(f"R:R={rr_data['rr_ratio']:.1f} — acceptable risk/reward"); bull_score += 1

    return {
        "ticker":        sym,
        "last_close":    px,
        "ret_1d":        r1,
        "ret_5d":        r5,
        "ret_20d":       r20,
        "ret_1y":        r252,
        "volatility":    volatility,
        "max_drawdown":  max_dd,
        "rsi":           rsi,
        "macd_hist":     macd_hist,
        "ema20":         ema20,
        "ema50":         ema50,
        "ema200":        ema200,
        "bb_pct":        bb_pct,
        "bb_width":      bb_width,
        "ml_pct":        ml_pct,
        "news_score":    news_score,
        "patterns":      patterns,
        "vol_ratio":     vol_ratio,
        "vol_quality":   vol_quality,
        "win_rate":      win_rate,
        "avg_yr_ret":    avg_yr_ret,
        "best_yr":       best_yr,
        "worst_yr":      worst_yr,
        "yearly":        yearly.to_dict(),
        "composite":     total,
        # Risk-Reward
        "rr_data":       rr_data,
        # Market regime
        "market_regime": market_regime.get("regime", "Unknown"),
        "mkt_multiplier": mkt_multiplier,
        # Sector
        "sector_data":   sector_data,
        "sector_boost":  sector_boost,
        # Weighted composite (new 5-component system)
        "weighted_score":      weighted["weighted_score"],
        "conviction_label":    weighted["conviction_label"],
        "conviction_reasons":  weighted["reasons"],
        "score_breakdown":     weighted["breakdown"],
        # screener-specific
        "bear_score":    bear_score,
        "bear_reasons":  bear_reasons,
        "bull_score":    bull_score,
        "bull_reasons":  bull_reasons,
    }


def deep_screen_stocks(tickers: list = None,
                        news_api_key: str = "",
                        max_workers: int = 6) -> tuple[list, list]:
    """
    Run full 10-year deep analysis on all tickers in parallel.
    Returns (bullish_list, bearish_list) sorted by score descending.
    """
    if tickers is None:
        tickers = WATCHLIST

    results = []

    def safe(sym):
        try:
            return _analyse_one(sym, news_api_key)
        except Exception as e:
            print(f"[screener] {sym} failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(safe, sym): sym for sym in tickers}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    bullish = sorted([r for r in results if r["bull_score"] >= 3],
                     key=lambda x: x["composite"], reverse=True)
    bearish = sorted([r for r in results if r["bear_score"] >= 2],
                     key=lambda x: x["bear_score"], reverse=True)
    return bullish, bearish


# ── Legacy screener (kept for backward compatibility) ────────

def screen_bearish(lookback_days: int = 2) -> list:
    """Wrapper kept so existing app.py import doesn't break."""
    _, bearish = deep_screen_stocks()
    return [
        {
            "ticker":         r["ticker"],
            "last_close":     r["last_close"],
            "change_pct":     r["ret_1d"] if lookback_days == 1 else r["ret_5d"],
            "rsi":            r["rsi"],
            "signal_reasons": r["bear_reasons"],
            "risk_level":     "🔴 High" if r["bear_score"] >= 5 else "🟡 Medium",
            "score":          r["bear_score"],
        }
        for r in bearish
    ]


# ── NEW: Bullish screener wrapper ────────────────────────────

def screen_bullish(lookback_days: int = 5) -> list:
    """
    Returns top bullish stocks from deep_screen_stocks.
    Checks 10-year data + ML signal + news sentiment.
    lookback_days: 1 = today, 5 = this week, 10 = next-week horizon.
    """
    bullish, _ = deep_screen_stocks()
    return [
        {
            "ticker":         r["ticker"],
            "last_close":     r["last_close"],
            "change_pct":     r["ret_1d"] if lookback_days == 1 else r["ret_5d"],
            "rsi":            r["rsi"],
            "signal_reasons": r["bull_reasons"],
            "confidence":     "🟢 Strong" if r["bull_score"] >= 6 else "🟡 Moderate",
            "score":          r["bull_score"],
            "composite":      r["composite"],
            "ml_pct":         r["ml_pct"],
            "news_score":     r["news_score"],
            "avg_yr_ret":     r["avg_yr_ret"],
            "win_rate":       r["win_rate"],
            "patterns":       r["patterns"],
            "volatility":     r["volatility"],
            "ret_1d":         r["ret_1d"],
            "ret_5d":         r["ret_5d"],
            "ret_20d":        r["ret_20d"],
        }
        for r in bullish
    ]


# ── NEW: Combined deep screener for both tabs ────────────────

def run_deep_screen(news_api_key: str = "") -> tuple:
    """
    Run one parallel deep scan and return (bullish_list, bearish_list)
    with full enriched data for both Bullish and Bearish screener tabs.
    Uses 10-year history, all indicators, ML signal, and news sentiment.
    """
    bullish_raw, bearish_raw = deep_screen_stocks(news_api_key=news_api_key)

    def _fmt_bull(r):
        return {
            "ticker":              r["ticker"],
            "last_close":          r["last_close"],
            "ret_1d":              r["ret_1d"],
            "ret_5d":              r["ret_5d"],
            "ret_20d":             r["ret_20d"],
            "rsi":                 r["rsi"],
            "signal_reasons":      r["bull_reasons"],
            "confidence":          "🟢 Strong" if r["bull_score"] >= 6 else "🟡 Moderate",
            "score":               r["bull_score"],
            "composite":           r["composite"],
            "ml_pct":              r["ml_pct"],
            "news_score":          r["news_score"],
            "avg_yr_ret":          r["avg_yr_ret"],
            "win_rate":            r["win_rate"],
            "patterns":            r["patterns"],
            "volatility":          r["volatility"],
            "best_yr":             r["best_yr"],
            "vol_ratio":           r.get("vol_ratio", 1.0),
            "vol_quality":         r.get("vol_quality", "N/A"),
            "rr_data":             r.get("rr_data", {}),
            "market_regime":       r.get("market_regime", "Unknown"),
            "sector_data":         r.get("sector_data", {}),
            "weighted_score":      r.get("weighted_score", 0),
            "conviction_label":    r.get("conviction_label", "N/A"),
            "conviction_reasons":  r.get("conviction_reasons", []),
            "score_breakdown":     r.get("score_breakdown", {}),
        }

    def _fmt_bear(r):
        return {
            "ticker":              r["ticker"],
            "last_close":          r["last_close"],
            "ret_1d":              r["ret_1d"],
            "ret_5d":              r["ret_5d"],
            "change_pct":          r["ret_5d"],
            "rsi":                 r["rsi"],
            "signal_reasons":      r["bear_reasons"],
            "risk_level":          "🔴 High" if r["bear_score"] >= 5 else "🟡 Medium",
            "score":               r["bear_score"],
            "composite":           r["composite"],
            "ml_pct":              r["ml_pct"],
            "news_score":          r["news_score"],
            "avg_yr_ret":          r["avg_yr_ret"],
            "worst_yr":            r["worst_yr"],
            "patterns":            r["patterns"],
            "volatility":          r["volatility"],
            "vol_ratio":           r.get("vol_ratio", 1.0),
            "vol_quality":         r.get("vol_quality", "N/A"),
            "market_regime":       r.get("market_regime", "Unknown"),
            "sector_data":         r.get("sector_data", {}),
            "weighted_score":      r.get("weighted_score", 0),
            "conviction_label":    r.get("conviction_label", "N/A"),
            "conviction_reasons":  r.get("conviction_reasons", []),
        }

    return (
        [_fmt_bull(r) for r in bullish_raw],
        [_fmt_bear(r) for r in bearish_raw],
    )