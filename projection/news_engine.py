"""
news_engine.py — News fetching + AI sentiment analysis

Pipeline:
  1. Fetch recent headlines via yfinance (free, no key required)
  2. FinBERT sentiment on all headlines (local model, finance-tuned)
  3. Claude API (claude-haiku) for high-impact events only:
       earnings, guidance, mergers, FDA decisions, SEC actions

Set ANTHROPIC_API_KEY in environment to enable Claude analysis.
FinBERT is downloaded automatically on first use (~440 MB).

Usage:
    from projection.news_engine import analyze_news
    result = analyze_news("AAPL", days_back=7)
"""

import os
import json
import math
import warnings
from datetime import datetime, timedelta


def _log(msg: str) -> None:
    """Print that survives Windows stdout quirks inside Streamlit."""
    try:
        print(msg, flush=True)
    except OSError:
        try:
            import sys
            sys.stderr.write(msg + "\n")
        except Exception:
            pass

HIGH_IMPACT_CATEGORIES = {"earnings", "guidance", "merger", "fda", "sec", "bankruptcy", "dividend"}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings":   ["earnings", "eps", "beat", "miss", "quarterly", "revenue", "profit", "q1", "q2", "q3", "q4", "results"],
    "guidance":   ["guidance", "outlook", "forecast", "raised", "lowered", "updated", "reaffirm"],
    "merger":     ["merger", "acquisition", "takeover", "buyout", "acquire", "merge", "deal", "bid"],
    "fda":        ["fda", "approval", "approved", "clinical", "trial", "drug", "phase", "nda", "bla"],
    "sec":        ["sec", "investigation", "fraud", "lawsuit", "settlement", "subpoena", "probe"],
    "bankruptcy": ["bankruptcy", "chapter 11", "restructuring", "default", "insolvency"],
    "dividend":   ["dividend", "payout", "yield", "distribution", "cut", "suspend"],
}

# lazy-loaded singletons
_finbert = None
_anthropic_client = None


# ── model loaders ──────────────────────────────────────────────────────────────

def _get_finbert():
    global _finbert
    if _finbert is not None:
        return _finbert if _finbert != "unavailable" else None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline

        _tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()

        _finbert = hf_pipeline(
            "text-classification",
            model=_model,
            tokenizer=_tok,
            truncation=True,
            max_length=512,
            device=-1,          # -1 = CPU in transformers
        )
    except Exception as e:
        _log(f"  [news] FinBERT load failed: {e}")
        _finbert = "unavailable"

    return _finbert if _finbert != "unavailable" else None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client if _anthropic_client != "unavailable" else None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _anthropic_client = "unavailable"
        return None

    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        _log("  [news] anthropic package not installed — Claude analysis disabled")
        _anthropic_client = "unavailable"

    return _anthropic_client if _anthropic_client != "unavailable" else None


# ── category detection ─────────────────────────────────────────────────────────

def _classify_category(headline: str) -> str:
    hl = headline.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in hl for kw in keywords):
            return cat
    return "general"


# ── FinBERT ────────────────────────────────────────────────────────────────────

def _finbert_batch(texts: list[str]) -> list[dict]:
    pipe = _get_finbert()
    if pipe is None:
        return [{"label": "neutral", "score": 0.5}] * len(texts)
    try:
        results = pipe(texts, batch_size=8)
        return [{"label": r["label"].lower(), "score": r["score"]} for r in results]
    except Exception as e:
        _log(f"  [news] FinBERT error: {e}")
        return [{"label": "neutral", "score": 0.5}] * len(texts)


_LABEL_TO_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def _finbert_signed_score(label: str, confidence: float) -> float:
    return _LABEL_TO_SCORE.get(label, 0.0) * confidence


# ── Claude analysis ────────────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Collapse whitespace + lowercase for deduplication."""
    return " ".join((title or "").lower().split())


def _claude_analyze(
    ticker: str,
    headline: str,
    summary: str = "",
    *,
    model: str,
) -> dict:
    client = _get_anthropic()
    if client is None:
        return {"score": 0.0, "sentiment": "neutral", "reasoning": "Claude unavailable"}

    body = headline
    if summary:
        body += f"\n\n{summary[:600]}"

    prompt = f"""You are a financial analyst. Analyze this news for stock {ticker}.
Respond with JSON only — no other text.

News: {body}

JSON format:
{{
  "sentiment": "positive"|"negative"|"neutral",
  "score": <float -1.0 to +1.0>,
  "impact_horizon": "short_term"|"medium_term"|"long_term",
  "reasoning": "<one sentence>"
}}"""

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": 0.0, "sentiment": "neutral", "reasoning": f"error: {e}"}


# ── news fetching ──────────────────────────────────────────────────────────────

def fetch_news(ticker: str, days_back: int = 7) -> list[dict]:
    """Fetch recent articles from yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        raw_news = tk.news or []
    except Exception as e:
        _log(f"  [news] Fetch error: {e}")
        return []

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    for item in raw_news[:25]:
        pub_ts = item.get("providerPublishTime", 0)
        pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else datetime.now()
        if pub_dt < cutoff:
            continue

        # yfinance returns nested content dict in newer versions
        content = item.get("content") or {}
        title   = item.get("title") or content.get("title") or ""
        summary = item.get("summary") or content.get("summary") or ""

        if not title:
            continue

        articles.append({
            "title":        title,
            "summary":      summary[:300],
            "publisher":    item.get("publisher") or (content.get("provider") or {}).get("displayName") or "",
            "published_at": pub_dt.isoformat(),
        })

    return articles


# ── main pipeline ──────────────────────────────────────────────────────────────

def analyze_news(ticker: str, days_back: int = 7) -> dict:
    """
    Full news analysis pipeline.

    Returns:
        available       bool
        n_articles      int
        sentiment_score float  -1..+1 aggregate
        signal          str    BULLISH / LEAN_BULLISH / NEUTRAL / ...
        articles        list   per-article analysis
        summary         str    human-readable summary
    """
    from projection_settings import load_projection_settings

    sett = load_projection_settings()
    claude_model = sett.anthropic_model

    articles = fetch_news(ticker, days_back)
    seen: set[str] = set()
    deduped: list[dict] = []
    for a in articles:
        key = _normalize_title(a.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    articles = deduped

    if not articles:
        return {
            "available":       False,
            "n_articles":      0,
            "sentiment_score": 0.0,
            "signal":          "NEUTRAL",
            "articles":        [],
            "summary":         "No recent news found",
        }

    headlines = [a["title"] for a in articles]
    fb_results = _finbert_batch(headlines)

    analyzed = []
    for article, fb in zip(articles, fb_results):
        category = _classify_category(article["title"])
        entry = {
            **article,
            "category":      category,
            "finbert_label": fb["label"],
            "finbert_conf":  round(fb["score"], 3),
        }

        if category in HIGH_IMPACT_CATEGORIES:
            cl = _claude_analyze(
                ticker, article["title"], article.get("summary", ""), model=claude_model
            )
            entry["claude_score"]     = round(float(cl.get("score", 0.0)), 3)
            entry["claude_sentiment"] = cl.get("sentiment", "neutral")
            entry["claude_reasoning"] = cl.get("reasoning", "")
            entry["impact_horizon"]   = cl.get("impact_horizon", "medium_term")
            entry["final_score"]      = entry["claude_score"]   # Claude wins for high-impact
        else:
            entry["final_score"]    = round(_finbert_signed_score(fb["label"], fb["score"]), 3)
            entry["impact_horizon"] = "short_term"

        analyzed.append(entry)

    # Weighted aggregate: high-impact × 2, recent × half-life decay (projection_settings)
    now = datetime.now()
    half_life_h = max(float(sett.news_decay_half_life_hours), 1.0)
    total_weight = 0.0
    weighted_sum = 0.0

    for a in analyzed:
        cat_weight = 2.0 if a["category"] in HIGH_IMPACT_CATEGORIES else 1.0

        try:
            age_hours = (now - datetime.fromisoformat(a["published_at"])).total_seconds() / 3600
        except Exception:
            age_hours = 24.0
        time_weight = math.exp(-math.log(2) * age_hours / half_life_h)

        weight = cat_weight * time_weight
        weighted_sum += a["final_score"] * weight
        total_weight += weight

    avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if avg_score > 0.30:
        signal = "BULLISH"
    elif avg_score > 0.10:
        signal = "LEAN_BULLISH"
    elif avg_score < -0.30:
        signal = "BEARISH"
    elif avg_score < -0.10:
        signal = "LEAN_BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "available":       True,
        "n_articles":      len(analyzed),
        "sentiment_score": round(avg_score, 3),
        "signal":          signal,
        "articles":        analyzed,
        "summary":         _make_summary(analyzed),
        "finbert_active":  _finbert not in (None, "unavailable"),
        "claude_active":   _anthropic_client not in (None, "unavailable"),
        "decay_half_life_hours": half_life_h,
        "claude_model":    claude_model,
    }


def _make_summary(articles: list[dict]) -> str:
    high = [a for a in articles if a["category"] in HIGH_IMPACT_CATEGORIES]
    if high:
        titles = "; ".join(a["title"][:80] for a in high[:3])
        return f"{len(articles)} articles — key events: {titles}"
    return f"{len(articles)} recent articles analyzed"


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Analyzing news for {ticker}...")
    result = analyze_news(ticker)
    print(f"Signal: {result['signal']}  Score: {result['sentiment_score']:+.3f}")
    print(f"Articles: {result['n_articles']}")
    print(f"Summary: {result['summary']}")
    for a in result["articles"][:5]:
        print(f"  [{a['category']}] {a['title'][:80]}  → {a['final_score']:+.2f}")
