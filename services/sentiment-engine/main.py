import os
import time
import json
import aiohttp
from fastapi import FastAPI
from shared.schemas import SentimentResponse

app = FastAPI(title="Sentiment Engine", version="1.0.0")
_cache: dict[str, tuple[float, SentimentResponse]] = {}
TELEGRAM_SENTIMENT_BOT_TOKEN = os.getenv("TELEGRAM_SENTIMENT_BOT_TOKEN", "").strip()
TELEGRAM_SENTIMENT_CHANNELS = {
    value.strip().lstrip("@").lower()
    for value in os.getenv("TELEGRAM_SENTIMENT_CHANNELS", "").split(",")
    if value.strip()
}
TELEGRAM_SENTIMENT_MAX_MESSAGES = int(os.getenv("TELEGRAM_SENTIMENT_MAX_MESSAGES", "50"))


async def _fetch_telegram_signal(session: aiohttp.ClientSession, symbol: str) -> float | None:
    """Read recent channel posts from a dedicated Telegram bot, when configured.

    A separate bot token is intentional: Telegram long polling allows only one
    consumer per token, so this must not share the emergency-control token.
    """
    if not TELEGRAM_SENTIMENT_BOT_TOKEN:
        return None
    endpoint = f"https://api.telegram.org/bot{TELEGRAM_SENTIMENT_BOT_TOKEN}/getUpdates"
    try:
        async with session.get(endpoint, params={
            "limit": TELEGRAM_SENTIMENT_MAX_MESSAGES,
            "allowed_updates": json.dumps(["channel_post", "message"]),
        }) as response:
            payload = await response.json(content_type=None)
        if not payload.get("ok"):
            return None
        positive = ("approval", "approved", "partnership", "listing", "adoption", "etf", "upgrade", "accumulate")
        negative = ("hack", "exploit", "lawsuit", "ban", "delist", "liquidation", "fraud", "breach", "dump")
        values: list[int] = []
        for update in payload.get("result", []):
            message = update.get("channel_post") or update.get("message") or {}
            chat = message.get("chat") or {}
            identity = str(chat.get("username") or chat.get("title") or "").lower().lstrip("@")
            if TELEGRAM_SENTIMENT_CHANNELS and identity not in TELEGRAM_SENTIMENT_CHANNELS:
                continue
            text = str(message.get("text") or message.get("caption") or "").lower()
            if symbol.lower() not in text and not any(word in text for word in ("bitcoin", "crypto", "market")):
                continue
            values.append(sum(word in text for word in positive) - sum(word in text for word in negative))
        return max(-1.0, min(1.0, (sum(values) / len(values)) * .25)) if values else None
    except Exception:
        return None


async def _fetch_sentiment(symbol: str) -> SentimentResponse:
    timeout = aiohttp.ClientTimeout(total=4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        fear_greed = None
        try:
            async with session.get("https://api.alternative.me/fng/?limit=1") as response:
                payload = await response.json()
                fear_greed = int(payload["data"][0]["value"])
        except Exception:
            pass

        score = 0.0
        source = "fallback"
        token = os.getenv("CRYPTOPANIC_API_KEY", "")
        cryptopanic_ok = False
        if token:
            try:
                async with session.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={"auth_token": token, "currencies": symbol,
                            "filter": "important", "public": "true"},
                ) as response:
                    data = await response.json()
                cryptopanic_ok = True
                positive = ("approval", "approved", "partnership", "listing", "adoption", "etf", "upgrade")
                negative = ("hack", "exploit", "lawsuit", "ban", "delist", "liquidation", "fraud")
                posts = data.get("results", [])[:30]
                values = [sum(w in str(post.get("title", "")).lower() for w in positive) -
                          sum(w in str(post.get("title", "")).lower() for w in negative) for post in posts]
                if values:
                    score = max(-1.0, min(1.0, sum(values) / max(len(values), 1) * .25))
                source = "cryptopanic"
            except Exception:
                pass
        # Cloudflare often 403s datacenter IPs on api/v1 — fall back to the
        # public RSS endpoint (no auth, accessible) when the API fails.
        if not cryptopanic_ok:
            try:
                import re as _re
                import xml.etree.ElementTree as ET
                async with session.get(
                    f"https://cryptopanic.com/news/rss/{symbol}/1/",
                    headers={"User-Agent": "Mozilla/5.0"},
                ) as response:
                    rss_body = await response.text()
                if response.status == 200 and rss_body.strip():
                    titles = []
                    try:
                        root = ET.fromstring(rss_body)
                        titles = [item.findtext("title", "").strip()
                                  for item in root.iter("item") if item.findtext("title", "").strip()]
                    except ET.ParseError:
                        # Malformed XML tolerated: regex extract <title> tags.
                        titles = [t.strip() for t in
                                  _re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                                              rss_body, _re.DOTALL)][:30]
                    if titles:
                        positive = ("approval", "approved", "partnership", "listing", "adoption", "etf", "upgrade")
                        negative = ("hack", "exploit", "lawsuit", "ban", "delist", "liquidation", "fraud")
                        values = [sum(w in t.lower() for w in positive) -
                                  sum(w in t.lower() for w in negative) for t in titles[:30]]
                        score = max(-1.0, min(1.0, sum(values) / max(len(values), 1) * .25))
                        source = "cryptopanic-rss"
                        cryptopanic_ok = True
            except Exception:
                pass
        if fear_greed is not None:
            source = "fear-greed+cryptopanic" if cryptopanic_ok else "fear-greed"
            score = max(-1.0, min(1.0, score + (fear_greed - 50) / 100)) / 2
        telegram_score = await _fetch_telegram_signal(session, symbol)
        if telegram_score is not None:
            score = (score + telegram_score) / 2 if source != "fallback" else telegram_score
            source = f"{source}+telegram"
        label = "bullish" if score > .15 else ("bearish" if score < -.15 else "neutral")
        fresh = fear_greed is not None or cryptopanic_ok or telegram_score is not None
        return SentimentResponse(pair=symbol, score=score, label=label,
                                 fear_greed=fear_greed, confidence=min(abs(score), 1.0),
                                 source=source, stale=not fresh)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "sentiment-engine"}

@app.get("/sentiment/{pair}", response_model=SentimentResponse)
async def sentiment(pair: str):
    symbol = pair.upper().split("/")[0].split(":")[0]
    cached = _cache.get(symbol)
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    try:
        result = await _fetch_sentiment(symbol)
    except Exception:
        result = SentimentResponse(pair=symbol, source="fallback", stale=True)
    _cache[symbol] = (time.time(), result)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
