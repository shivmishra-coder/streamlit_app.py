# ============================================================
# debug_yfinance.py
# Run this FIRST: python debug_yfinance.py
# It will tell you exactly what is wrong and print the fix.
# ============================================================

import sys
import subprocess

print("=" * 60)
print("StockIQ — yfinance Diagnostic Tool")
print("=" * 60)

# ── Step 1: Check Python version ─────────────────────────────
print(f"\n[1] Python version: {sys.version}")

# ── Step 2: Check yfinance version ───────────────────────────
try:
    import yfinance as yf
    print(f"[2] yfinance version: {yf.__version__}")
except ImportError:
    print("[2] ❌ yfinance NOT installed! Run: pip install yfinance==0.2.54")
    sys.exit(1)

# ── Step 3: Check pandas version ─────────────────────────────
try:
    import pandas as pd
    print(f"[3] pandas version: {pd.__version__}")
except ImportError:
    print("[3] ❌ pandas NOT installed!")
    sys.exit(1)

# ── Step 4: Test actual download ─────────────────────────────
print("\n[4] Testing yfinance download for TATAMOTORS.NS ...")
try:
    df = yf.download(
        "TATAMOTORS.NS",
        start="2024-01-01",
        end="2025-01-01",
        progress=False,
        auto_adjust=True,
    )
    print(f"    Shape     : {df.shape}")
    print(f"    Empty     : {df.empty}")
    print(f"    Col type  : {type(df.columns)}")
    print(f"    Columns   : {df.columns.tolist()}")
    if not df.empty:
        print(f"    First row : {df.iloc[0].to_dict()}")
        print("\n✅ Download WORKS — the issue is column parsing only.")
    else:
        print("\n⚠️  Download returned EMPTY DataFrame.")
        print("    Possible reasons:")
        print("    - No internet connection")
        print("    - Yahoo Finance blocking your IP / region")
        print("    - yfinance version mismatch")
except Exception as e:
    print(f"    ❌ Download FAILED with error: {e}")

# ── Step 5: Test Ticker.history() ────────────────────────────
print("\n[5] Testing Ticker.history() ...")
try:
    t   = yf.Ticker("TATAMOTORS.NS")
    df2 = t.history(period="1mo")
    print(f"    Shape  : {df2.shape}")
    print(f"    Columns: {df2.columns.tolist()}")
    if not df2.empty:
        print("✅ Ticker.history() WORKS")
    else:
        print("⚠️  Ticker.history() also empty")
except Exception as e:
    print(f"❌ Ticker.history() FAILED: {e}")

# ── Step 6: Check installed versions ─────────────────────────
print("\n[6] Installed package versions:")
packages = ["yfinance", "pandas", "numpy", "streamlit",
            "scikit-learn", "plotly", "requests"]
for pkg in packages:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Version:"):
                print(f"    {pkg}: {line.split(': ')[1]}")
                break
    except Exception:
        print(f"    {pkg}: could not check")

print("\n" + "=" * 60)
print("Copy the output above and share it to get a precise fix.")
print("=" * 60)