from __future__ import annotations

import json
import math
import os
import re

import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.supreme_final import tfidf_decay_sentiment

app = FastAPI(title="News Alpha", version="2.0.0 (TF-IDF & Decay Edition)")
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
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)