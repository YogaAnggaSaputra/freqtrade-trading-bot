#!/usr/bin/env python3
import json
import os

config = {
    "exportdir": "/freqtrade/user_data/trades",
    "strategy_path": "/freqtrade/configs/strategies/",
    "data_format": "feather",
    "datadir": "/freqtrade/user_data/data",
    "process_only_new_candles": False,
    "strategy": "AITradingStrategy",
    "exchange": {
        "name": "binanceusdm",
        "key": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", ""),
        "pair_whitelist": [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
            "SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
            "DOGE/USDT:USDT", "DOT/USDT:USDT", "AVAX/USDT:USDT",
            "MATIC/USDT:USDT", "LINK/USDT:USDT", "UNI/USDT:USDT",
            "LTC/USDT:USDT", "ATOM/USDT:USDT", "APE/USDT:USDT",
            "SAND/USDT:USDT", "MANA/USDT:USDT", "AXS/USDT:USDT",
            "SHIB/USDT:USDT", "FTM/USDT:USDT"
        ],
        "symmetrize": True,
        "ccxt_async_config": {"enableRateLimit": True}
    },
    "dry_run": False,
    "stake_amount": "unlimited",
    "stake_currency": "USDT",
    "max_open_trades": 5,
    "timeframe": "5m",
    "minimal_roi": {"0": 0.1},
    "stoploss": -0.02,
    "trailing_stop": False,
    "startup_candle_count": 30,
    "enable_protections": ["StoplossGuard", "CooldownPeriod", "MaxDrawdown"],
    "pairlists": [
        {"method": "StaticPairList"},
        {"method": "AgeFilter", "min_days_listed": 1}
    ]
}

with open("/tmp/test_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Config created at /tmp/test_config.json")
print(f"Pairs: {len(config['exchange']['pair_whitelist'])}")
print(f"API Key: {config['exchange']['key'][:10]}...")
