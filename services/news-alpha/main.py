from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import aiohttp
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.supreme_final import tfidf_decay_sentiment

logger = logging.getLogger(__name__)

app = FastAPI(title="News Alpha", version="3.0.0 (TF-IDF & Decay + RSS Poller)")
class Headline(BaseModel):
    pair: str
    headline: str
    elapsed_minutes: float = 0.0

POS_WEIGHTS = {"approval": 0.5, "approved": 0.5, "partnership": 0.4, "listing": 0.4, "launch": 0.3, "adoption": 0.4, "etf": 0.6, "upgrade": 0.3}
NEG_WEIGHTS = {"hack": 0.6, "exploit": 0.6, "lawsuit": 0.5, "ban": 0.5, "delist": 0.5, "liquidation": 0.4, "fraud": 0.6, "breach": 0.5}


def _tfidf_classify(item: Headline) -> dict:
    score = tfidf_decay_sentiment(
        headline=item.headline,
        pos_weights=POS_WEIGHTS,
        neg_weights=NEG_WEIGHTS,
        elapsed_minutes=item.elapsed_minutes,
        half_life_min=60.0,
    )
    return {
        "pair": item.pair.upper(),
        "score": score,
        "label": "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral"),
        "event": "negative_event" if score < -0.20 else ("positive_event" if score > 0.20 else "none"),
        "source": "tfidf_decay",
    }


def _parse_json_response(value: str) -> dict | None:
    """Parse strict JSON plus common fenced/verbose local-model responses."""
    value = value.strip()
    candidates = [value]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    if "{" in value and "}" in value:
        candidates.append(value[value.find("{"):value.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


NEWS_LLM_ENABLED = os.getenv("NEWS_LLM_ENABLED", "false").strip().lower() == "true"
NEWS_LLM_API_KEY = os.getenv("NEWS_LLM_API_KEY", "").strip()
NEWS_LLM_URL = os.getenv("NEWS_LLM_URL", "http://ollama:11434/api/generate")
NEWS_LLM_MODEL = os.getenv("NEWS_LLM_MODEL", "qwen2.5:1.5b")
NEWS_LLM_TIMEOUT_SECONDS = float(os.getenv("NEWS_LLM_TIMEOUT_SECONDS", "2.5"))


async def _llm_classify(item: Headline) -> dict | None:
    """Optional Ollama/OpenAI-compatible classifier; never fail-closed on news outage."""
    if not NEWS_LLM_ENABLED or not item.headline.strip():
        return None
    prompt = (
        "Classify this crypto market headline. Return JSON only with exactly these keys: "
        "score (number -1 to 1), label (bullish/bearish/neutral), "
        "event (positive_event/negative_event/none). Do not add markdown.\n"
        f"Pair: {item.pair}\nHeadline: {item.headline}"
    )
    try:
        timeout = aiohttp.ClientTimeout(total=NEWS_LLM_TIMEOUT_SECONDS)
        headers = {"Content-Type": "application/json"}
        if NEWS_LLM_API_KEY:
            headers["Authorization"] = f"Bearer {NEWS_LLM_API_KEY}"
        # OpenAI-compatible payload; falls back to Ollama format if URL contains /api/generate
        is_ollama = "/api/generate" in NEWS_LLM_URL
        if is_ollama:
            payload = {"model": NEWS_LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"}
        else:
            payload = {
                "model": NEWS_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "response_format": {"type": "json_object"},
            }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(NEWS_LLM_URL, json=payload, headers=headers) as response:
                if response.status >= 400:
                    return None
                payload = await response.json(content_type=None)
        raw = payload.get("response")
        if raw is None and isinstance(payload.get("choices"), list) and payload["choices"]:
            raw = payload["choices"][0].get("message", {}).get("content")
        parsed = _parse_json_response(str(raw or ""))
        if not parsed:
            return None
        score = float(parsed.get("score"))
        if not math.isfinite(score):
            return None
        score = max(-1.0, min(1.0, score))
        label = str(parsed.get("label", "neutral")).lower()
        event = str(parsed.get("event", "none")).lower()
        if label not in {"bullish", "bearish", "neutral"}:
            label = "bullish" if score > .15 else ("bearish" if score < -.15 else "neutral")
        if event not in {"positive_event", "negative_event", "none"}:
            event = "none"
        return {
            "pair": item.pair.upper(), "score": score, "label": label,
            "event": event, "source": f"local_llm:{NEWS_LLM_MODEL}",
        }
    except Exception:
        return None


@app.get("/health")
async def health(): return {"status": "healthy", "service": "news-alpha"}
@app.post("/classify")
async def classify(item: Headline):
    return await _llm_classify(item) or _tfidf_classify(item)


# ── RSS Feed Poller ──

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://cryptopanic.com/news/rss/BTC/1/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]
_latest_news: list[dict] = []
_news_lock = asyncio.Lock()
POLL_INTERVAL = int(os.getenv("NEWS_POLL_INTERVAL_SECONDS", "300"))


async def _fetch_rss(session: aiohttp.ClientSession, url: str) -> list[dict]:
    """Fetch RSS, parse XML, return [{title, link, pub_date, source}]."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15),
                                headers={"User-Agent": "Mozilla/5.0"}) as r:
            if r.status >= 400:
                return []
            body = await r.text()
    except Exception:
        return []
    articles = []
    source_label = url.split("//")[1].split("/")[0].replace("www.", "")
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(body)
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub = item.findtext("pubDate", "") or item.findtext("dc:date", "", ns)
            if not title:
                continue
            articles.append({
                "title": title, "link": link, "pub_date": pub,
                "source": source_label, "fetched": datetime.now(UTC).isoformat(),
            })
    except ET.ParseError:
        pass
    return articles


async def _poller_loop():
    """Periodic RSS poller: fetch, classify, store 5 latest per pair."""
    global _latest_news
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                all_articles = []
                for feed in RSS_FEEDS:
                    all_articles.extend(await _fetch_rss(session, feed))
                if not all_articles:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                seen = set()
                unique = []
                for a in all_articles:
                    key = a["title"].lower().strip()
                    if key not in seen:
                        seen.add(key)
                        unique.append(a)
                unique.sort(key=lambda x: x.get("pub_date", ""), reverse=True)
                classified = []
                for a in unique[:10]:
                    item = Headline(pair="BTC/USDT", headline=a["title"])
                    result = await _llm_classify(item) or _tfidf_classify(item)
                    a["score"] = result.get("score", 0)
                    a["label"] = result.get("label", "neutral")
                    classified.append(a)
                async with _news_lock:
                    _latest_news = classified[:10]
                    logger.info("News poller: %d articles (%d unique, %d classified)",
                                len(all_articles), len(unique), len(classified))
        except Exception as exc:
            logger.debug("News poller error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_poller_loop())


@app.get("/latest")
async def latest():
    """Return the 5 most recent classified articles."""
    async with _news_lock:
        return {"articles": _latest_news[:5]}


if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)