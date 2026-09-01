import asyncio
import os
import aiohttp

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RISK_URL = os.getenv("RISK_GATEWAY_URL", "http://risk-gateway:8000")
KILL_SWITCH_PIN = os.getenv("KILL_SWITCH_PIN", "")
ENABLED = os.getenv("TELEGRAM_CONTROL_ENABLED", "false").lower() == "true"

async def poll():
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    offset = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
        while True:
            try:
                async with session.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                                       params={"timeout": 30, "offset": offset}) as response:
                    updates = (await response.json()).get("result", [])
                for update in updates:
                    offset = max(offset, int(update["update_id"]) + 1)
                    message = update.get("message", {})
                    if str(message.get("chat", {}).get("id")) != str(CHAT_ID): continue
                    command = str(message.get("text", "")).split()
                    if not command or command[0].lower() not in {"/killswitch", "/kill"}: continue
                    level = command[1].lower() if len(command) > 1 else "black"
                    if level not in {"yellow", "orange", "red", "black", "green"}: continue
                    async with session.post(f"{RISK_URL}/killswitch", json={"level": level, "reason": "telegram-control", "pin": KILL_SWITCH_PIN}) as result:
                        await result.read()
            except asyncio.CancelledError: raise
            except Exception:
                await asyncio.sleep(5)

async def main():
    if ENABLED: await poll()
    else: await asyncio.Event().wait()

if __name__ == "__main__": asyncio.run(main())
