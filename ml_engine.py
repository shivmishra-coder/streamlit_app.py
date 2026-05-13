# ============================================================
# ml_engine.py
# Random Forest model for next-day closing price prediction.
# Trains on 2 years of historical data with feature engineering.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ── Feature Engineering ──────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML features from OHLCV + indicator data.

    Features used:
      - Lagged close prices (1–5 days)
      - Rolling means (5, 10, 20 day)
      - Rolling std (5 day) — volatility proxy
      - Daily return
      - RSI (if present)
      - MACD (if present)
      - EMA_20, EMA_50 (if present)

    Returns:
        DataFrame with a 'Target' column = next day's Close.
    """
    feat = df.copy()

    # Lag features
    for lag in range(1, 6):
        feat[f"Lag_{lag}"] = feat["Close"].shift(lag)

    # Rolling statistics
    feat["Roll_Mean_5"]  = feat["Close"].rolling(5).mean()
    feat["Roll_Mean_10"] = feat["Close"].rolling(10).mean()
    feat["Roll_Mean_20"] = feat["Close"].rolling(20).mean()
    feat["Roll_Std_5"]   = feat["Close"].rolling(5).std()

    # Return
    feat["Daily_Return"] = feat["Close"].pct_change()

    # Target: next day close
    feat["Target"] = feat["Close"].shift(-1)

    feat.dropna(inplace=True)
    return feat


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature column names actually present in df."""
    candidates = (
        [f"Lag_{i}" for i in range(1, 6)]
        + ["Roll_Mean_5", "Roll_Mean_10", "Roll_Mean_20", "Roll_Std_5",
           "Daily_Return"]
        + ["RSI", "MACD", "MACD_Signal", "EMA_20", "EMA_50"]
    )
    return [c for c in candidates if c in df.columns]


# ── Model Training & Prediction ──────────────────────────────

def train_and_predict(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest Regressor and predict the next day's Close.

    Pipeline:
        1. Feature engineering on enriched OHLCV data
        2. 80/20 chronological train-test split (no shuffle — avoids leakage)
        3. StandardScaler normalisation
        4. RandomForestRegressor (200 trees, tuned hyper-params)
        5. MAE on hold-out test set for honest reporting
        6. Retrain on full data → predict next candle

    Args:
        df: Enriched DataFrame (output of data_engine.enrich_data)

    Returns:
        dict with keys:
            predicted_price (float)
            mae             (float)
            feature_importance (pd.Series)
            last_close      (float)
            direction       ('Up' | 'Down' | 'Flat')
    """
    try:
        feat_df = build_features(df)
        feature_cols = _get_feature_cols(feat_df)

        if len(feat_df) < 60:
            return {"error": "Not enough data to train the model (min 60 rows needed)."}

        X = feat_df[feature_cols].values
        y = feat_df["Target"].values

        # Chronological split — critical for time-series validity
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluation on hold-out
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)

        # Retrain on ALL data for the final prediction
        X_all = scaler.fit_transform(X)
        model.fit(X_all, y)

        # Latest row as input (using the last available features)
        latest_features = feat_df[feature_cols].iloc[-1].values.reshape(1, -1)
        latest_scaled   = scaler.transform(latest_features)
        predicted_price = float(model.predict(latest_scaled)[0])

        last_close = float(df["Close"].iloc[-1])
        diff       = predicted_price - last_close
        direction  = "Up" if diff > 0.01 else ("Down" if diff < -0.01 else "Flat")

        # Feature importances
        importance = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        return {
            "predicted_price":    round(predicted_price, 2),
            "mae":                round(mae, 2),
            "last_close":         round(last_close, 2),
            "direction":          direction,
            "feature_importance": importance,
        }

    except Exception as e:
        return {"error": str(e)}