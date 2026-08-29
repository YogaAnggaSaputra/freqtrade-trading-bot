#!/usr/bin/env python3
"""Freqtrade Runtime Service - Simplified"""
import os
import sys
import json
import threading
import uuid
from typing import Optional
import asyncio
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn
import logging
from urllib.request import urlopen
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Freqtrade Runtime API")
freqtrade_process: Optional[threading.Thread] = None
bot_running = False
bot_state = "stopped"

class StartRequest(BaseModel):
    mode: str | None = None


def adaptive_whitelist(fallback: list[str]) -> list[str]:
    """Resolve live pairlist once at startup; preserve fallback on outage."""
    if os.getenv("ADAPTIVE_WHITELIST_ENABLED", "false").lower() != "true":
        return fallback
    url = os.getenv("ADAPTIVE_WHITELIST_URL", "http://adaptive-whitelist:8000") + "/whitelist?limit=50"
    try:
        import json as _json
        with urlopen(url, timeout=3) as response:  # noqa: S310
            pairs = _json.loads(response.read().decode()).get("pairs", [])
        if pairs:
            logger.info("Adaptive whitelist selected %d live pairs", len(pairs))
            return pairs
    except Exception as exc:
        logger.warning("Adaptive whitelist unavailable, using local fallback: %s", exc)
    return fallback

def create_config(mode: str) -> str:
    # Read from secrets or environment
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    # Try reading from Docker secrets if available
    if not api_key or api_key == "changeme":
        try:
            with open("/run/secrets/binance_api_key", "r") as f:
                api_key = f.read().strip()
        except:
            pass
    if not api_secret or api_secret == "changeme":
        try:
            with open("/run/secrets/binance_api_secret", "r") as f:
                api_secret = f.read().strip()
        except:
            pass

    # [HARDEN] Fail-closed: jangan start dengan placeholder/kosong.
    # Live trading tanpa kredensial valid = berbahaya diam-diam.
    if not api_key or api_key == "changeme" or not api_secret or api_secret == "changeme":
        raise RuntimeError(
            "BINANCE_API_KEY/API_SECRET kosong atau masih placeholder 'changeme' — "
            "refuse to start (fail-closed). Cek .env / secrets/."
        )
    
    # Auto-generate pair whitelist dari feather files yang tersedia
    import glob
    import re
    feather_dir = "/freqtrade/user_data/data/futures"
    feather_files = glob.glob(os.path.join(feather_dir, "*-5m-futures.feather"))
    top_pairs = []
    seen = set()
    for fpath in feather_files:
        fname = os.path.basename(fpath)
        # Format: BTC_USDT-5m-futures.feather atau BTC_USDT_USDT-5m-futures.feather
        match = re.match(r'(.+?)(?:_USDT)?-5m-futures\.feather', fname)
        if match:
            base = match.group(1)
            # Remove trailing _USDT if present (from BTC_USDT_USDT)
            base = base.replace("_USDT_USDT", "").replace("_USDT", "")
            pair = f"{base}/USDT:USDT"
            if pair not in seen:
                seen.add(pair)
                top_pairs.append(pair)
    
    if not top_pairs:
        top_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
            "SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
        ]
    top_pairs = adaptive_whitelist(top_pairs)
    
    # Tinggal 1 posisi (max_open_trades=1) — cukup 200 pair liquid, bukan 527.
    # VolumePairList = pencarian pair by volume di exchange (native Freqtrade).
    # Analisis 527 pair ≈ 130s (kelewatan ambang 75s) → 200 pair ≈ 50s, aman.
    VOLUME_PAIR_LIMIT = 200
    logger.info(f"Generated whitelist with {len(top_pairs)} pairs from feather files (VolumePairList top {VOLUME_PAIR_LIMIT})")
    
    config = {
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "stake_currency": "USDT",
        "stake_amount": 1,
        "tradable_balance_ratio": 0.99,
        "dry_run": mode != "live",
        # Dry-run wallet disamakan dgn balance real akun (~$7) supaya stake yang
        # dihitung freqtrade lolos limit exposure risk-gateway. Wallet $100 bikin
        # semua entry ditolak (exposure > limit) → gak ada trade buat verifikasi.
        "dry_run_wallet": 7,
        "max_open_trades": 1,
        "max_leverage": 5,
        "leverage": 5,
        # [PARTIAL-TP-OFF] partial TP (40/35/25) tidak feasible di stake mikro:
        # notional ~$5.34 dipecah 3 → tiap porsi <$5 min notional Binance → REJECT.
        # Full-position + trailing SL (Layer 1-4) yang nangkep upside. Flip True
        # lagi kalau stake per-trade ≥$2 @5x (notional ≥$20, tiap pecahan >$5).
        "position_adjustment_enable": os.getenv("POSITION_ADJUSTMENT_ENABLE", "false").lower() == "true",
        # DB disimpan di volume /freqtrade/user_data agar trade history
        # bertahan lintas rebuild container (sebelumnya di /app = ephemeral).
        "db_url": (
            "sqlite:////freqtrade/user_data/tradesv3.sqlite"
            if mode == "live"
            else "sqlite:////freqtrade/user_data/tradesv3.dryrun.sqlite"
        ),
        # Log ke file di volume agar bisa di-inspect setelah restart.
        "logfile": "/freqtrade/user_data/logs/freqtrade.log",
        "timeframe": "5m",
        "process_only_new_candles": True,
        "strategy": "AITradingStrategy",
        "strategy_path": "/freqtrade/configs/strategies/",
        "datadir": "/freqtrade/user_data/data",
        "data_format_ohlcv": "feather",
        "data_format_trades": "feather",
        "exportdir": "/freqtrade/user_data/trades",
        "user_data_dir": "/freqtrade/user_data",
        # [ROI-SAFETY-NET] ROI 0.10 terlalu ketat: jual full di +10% margin
        # (~+2% harga @5x) sebelum trailing SL Layer 1-4 sempat ngejar tren
        # (kasus MUBARAK exit +10% padahal harga lanjut naik). 0.50 = safety-net
        # darurat saja (profit >50% margin); exit utama diserahkan ke trailing SL.
        # HANYA key "0" statis — tanpa decay bertahap yang diam-diam mempersempit.
        "minimal_roi": {"0": 0.50},
        "stoploss": -0.02,
        "trailing_stop": False,
        "startup_candle_count": 200,
        "exchange": {
            "name": "binanceusdm",
            "key": api_key,
            "secret": api_secret,
            "pair_whitelist": top_pairs,
            "symmetrize": True,
            "ccxt_config": {"enableRateLimit": True},
            "ccxt_async_config": {"enableRateLimit": True}
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1
        },
        # Watchdog order entry/exit yang belum fill.
        # Tanpa ini, order entry bisa nggantung selamanya saat price_side
        # skip (mis. orderbook fetch timeout) — bot diam, modal idle.
        # Samakan dengan strategi: 5 menit entry, 15 menit exit, max 3 retry.
        "unfilledtimeout": {
            "entry": 5,
            "exit": 15,
            "exit_timeout_count": 3,
            "unit": "minutes",
        },
        "pairlists": [
            {
                "method": "VolumePairList",
                "number_assets": 200,
                "sort_key": "quoteVolume",
                "min_value": 0,
                "refresh_period": 1800
            }
        ],
        "bot_name": "aitrading_bot",
        "initial_state": "running",
        "force_entry": False,
        "internals": {
            "process_throttle_secs": 5
        }
    }
    
    config_path = f"/tmp/freqtrade_config_{uuid.uuid4().hex}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

def run_freqtrade(config_path: str):
    global freqtrade_process
    logger.info(f"Starting freqtrade with config: {config_path}")
    import subprocess
    cmd = [
        sys.executable, "-m", "freqtrade", "trade",
        "--config", config_path,
        "--strategy", "AITradingStrategy",
        "--strategy-path", "/freqtrade/configs/strategies/"
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    # stderr=STDOUT: gabungkan ke stdout yang memang dibaca di loop bawah.
    # Kalau stderr=PIPE tapi tidak pernah dibaca, buffer pipe penuh dan
    # proses freqtrade blocking (terlihat sebagai bot idle 0% CPU).
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    while proc.poll() is None:
        line = proc.stdout.readline()
        if line:
            logger.info(f"FREQTRADE: {line.strip()}")
    logger.info(f"Freqtrade exited with code {proc.returncode}")
    global bot_running, bot_state
    bot_running = False
    bot_state = "stopped" if proc.returncode == 0 else "error"
    freqtrade_process = None

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "freqtrade-runtime"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
async def status():
    return {"running": bot_running, "state": bot_state, "has_process": freqtrade_process is not None}

@app.post("/start")
async def start_bot(request: StartRequest):
    global freqtrade_process, bot_running, bot_state
    if bot_running:
        return {"status": "already_running"}
    try:
        mode = (request.mode or os.getenv("TRADE_MODE", "dry")).lower()
        if mode not in {"live", "dry", "dry_run", "demo", "testnet"}:
            raise ValueError("mode must be one of live, dry, dry_run, demo, testnet")
        config_path = create_config(mode)
        thread = threading.Thread(target=run_freqtrade, args=(config_path,), daemon=True)
        thread.start()
        freqtrade_process = thread
        bot_running = True
        bot_state = "running"
        await asyncio.sleep(2)
        return {"status": "started", "mode": mode, "config": config_path}
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        bot_state = "error"
        raise

@app.post("/stop")
async def stop_bot():
    global freqtrade_process, bot_running, bot_state
    if freqtrade_process is None:
        return {"status": "not_running"}
    os.system("pkill -f 'python.*freqtrade' 2>/dev/null")
    freqtrade_process.join(timeout=10)
    bot_running = False
    bot_state = "stopped"
    return {"status": "stopped"}

async def start_bot_internal(mode: str):
    """Start bot (used by /start endpoint and auto-start on boot)."""
    global freqtrade_process, bot_running, bot_state
    if bot_running:
        return
    config_path = create_config(mode)
    thread = threading.Thread(target=run_freqtrade, args=(config_path,), daemon=True)
    thread.start()
    freqtrade_process = thread
    bot_running = True
    bot_state = "running"
    await asyncio.sleep(2)


async def main():
    global bot_state
    logger.info("Starting Freqtrade Runtime Service...")
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    await asyncio.sleep(1)

    # Auto-start live bot on boot (TRADE_MODE=live). State is in-memory, so a
    # container restart would otherwise leave the bot stopped until manual /start.
    auto_mode = os.getenv("TRADE_MODE", "demo").lower()
    if auto_mode == "live":
        logger.info(f"TRADE_MODE=live -> auto-starting bot (mode={auto_mode})")
        try:
            await start_bot_internal("live")
            logger.info(f"Bot auto-started (state={bot_state})")
        except Exception as e:
            logger.error(f"Auto-start failed: {e}")
            bot_state = "error"
    else:
        logger.info(f"TRADE_MODE={auto_mode} -> bot stays stopped (manual /start required)")

    logger.info("Freqtrade Runtime Service is ready")
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
