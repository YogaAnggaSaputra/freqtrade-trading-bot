#!/usr/bin/env python3
"""Download OHLCV for top 20 pairs only - optimized"""
import requests
import pandas as pd
import pyarrow.feather as feather
import time
import os
import sys

# Top 20 pairs
pairs = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT",
    "LINKUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "APEUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "SHIBUSDT", "FTMUSDT"
]

base_dir = "/freqtrade/user_data/data/binanceusdm/futures/5m"
os.makedirs(base_dir, exist_ok=True)

print(f"Starting download for {len(pairs)} pairs...")
success, skipped, errors = 0, 0, 0

for i, symbol in enumerate(pairs):
    try:
        pair = symbol.replace("USDT", "USDT:USDT")
        coin = symbol.replace("USDT", "")
        
        # Get all candles
        all_candles = []
        start_time = int(time.time() - 14*24*60*60*1000)  # 14 days
        end_time = int(time.time())
        
        while start_time < end_time:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={start_time}&endTime={end_time}&limit=1500"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if not data:
                    break
                
                for candle in data:
                    all_candles.append({
                        "timestamp": pd.Timestamp(candle[0], unit="ms"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })
                
                start_time = data[-1][0] + 1
            else:
                print(f"  Failed: {symbol}")
                break
        
        if all_candles:
            df = pd.DataFrame(all_candles)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            
            dir_path = f"{base_dir}/{coin}"
            os.makedirs(dir_path, exist_ok=True)
            feather.write_feather(df, f"{dir_path}/{coin}_5m.feather")
            
            success += 1
            print(f"[{i+1}/{len(pairs)}] ✅ {coin}: {len(df)} candles")
        else:
            skipped += 1
            print(f"[{i+1}/{len(pairs)}] ⏭️ {coin}: No data")
        
        time.sleep(0.2)
        
    except Exception as e:
        errors += 1
        print(f"[{i+1}/{len(pairs)}] ❌ {symbol}: {str(e)[:50]}")
        time.sleep(1)

print(f"\n{'='*50}")
print(f"Download complete!")
print(f"✅ Success: {success}")
print(f"⏭️ Skipped: {skipped}")
print(f"❌ Errors: {errors}")
print(f"{'='*50}")
