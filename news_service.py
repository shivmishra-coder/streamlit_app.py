# ============================================================
# news_service.py
# NewsAPI integration + rule-based sentiment scoring.
# ============================================================

import os
from datetime import datetime, timedelta

# Guard import — newsapi is optional; app degrades gracefully
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False


# ── Sentiment Lexicon ────────────────────────────────────────
# Simple but effective word-list approach (no external NLP deps).
# Covers financial-domain vocabulary for better accuracy.

POSITIVE_WORDS = {
    "surge", "gain", "rally", "profit", "record", "growth", "rise",
    "bullish", "outperform", "beat", "upgrade", "strong", "positive",
    "soar", "jump", "boost", "win", "revenue", "dividend", "boom",
    "recovery", "high", "exceed", "upbeat", "optimistic", "buy",
    "expand", "opportunity", "success", "breakthrough", "promising",
}

NEGATIVE_WORDS = {
    "fall", "drop", "crash", "loss", "decline", "bearish", "underperform",
    "miss", "downgrade", "weak", "negative", "plunge", "slump", "sell",
    "concern", "risk", "warning", "debt", "bankruptcy", "fraud",
    "lawsuit", "investigation", "cut", "layoff", "recession", "crisis",
    "volatile", "uncertainty", "deficit", "penalty", "disappointing",
}


# ── Sentiment Scorer ─────────────────────────────────────────

def score_sentiment(text: str) -> dict:
    """
    Compute a simple sentiment score for a headline.

    Method:
        For each word in the text, check membership in POSITIVE_WORDS
        and NEGATIVE_WORDS sets. Net score = positives − negatives.

    Args:
        text: Headline or any short text string.

    Returns:
        dict with keys:
            label  ('Positive' | 'Negative' | 'Neutral')
            score  (int: net word count)
            pos    (int: positive word count)
            neg    (int: negative word count)
    """
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?\"'") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?\"'") in NEGATIVE_WORDS)
    net = pos - neg

    if net > 0:
        label = "Positive"
    elif net < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": net, "pos": pos, "neg": neg}


# ── News Fetcher ─────────────────────────────────────────────

def fetch_news(ticker: str, api_key: str, n: int = 5) -> list[dict]:
    """
    Fetch the latest n news articles for a ticker symbol.

    Strategy:
        Uses the ticker's base symbol (strips exchange suffix like '.NS')
        as the search query so results are more relevant.

    Args:
        ticker:  Full ticker (e.g. 'TATAMOTORS.NS')
        api_key: NewsAPI key string
        n:       Number of articles to return (default 5)

    Returns:
        List of article dicts, each with keys:
            title, description, url, published_at,
            source, sentiment (from score_sentiment)

    On any error returns a list with a single error-marker dict.
    """
    if not api_key or api_key == "your_newsapi_key_here":
        return [{"error": "NEWS_API_KEY not configured. Add it to your .env file."}]

    if not NEWSAPI_AVAILABLE:
        return [{"error": "newsapi-python not installed. Run: pip install newsapi-python"}]

    # Strip exchange suffix for cleaner search queries
    base_symbol = ticker.split(".")[0]

    try:
        client = NewsApiClient(api_key=api_key)
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        response = client.get_everything(
            q=base_symbol,
            language="en",
            sort_by="publishedAt",
            page_size=n,
            from_param=from_date,
        )

        articles = response.get("articles", [])
        results  = []

        for art in articles[:n]:
            headline = art.get("title", "No title")
            sentiment = score_sentiment(headline)
            results.append({
                "title":        headline,
                "description":  art.get("description", ""),
                "url":          art.get("url", "#"),
                "published_at": art.get("publishedAt", ""),
                "source":       art.get("source", {}).get("name", "Unknown"),
                "sentiment":    sentiment,
            })

        if not results:
            return [{"error": f"No news found for '{base_symbol}' in the past 7 days."}]

        return results

    except Exception as e:
        return [{"error": f"NewsAPI error: {str(e)}"}]