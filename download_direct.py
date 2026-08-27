#!/usr/bin/env python3
"""Download OHLCV data directly from Binance API for all pairs"""
import requests
import pandas as pd
import pyarrow.feather as feather
import time
import os
from datetime import datetime, timedelta

# Create data directory
base_dir = "/freqtrade/user_data/data/futures"
os.makedirs(base_dir, exist_ok=True)

# Get all pairs from Binance
print("Fetching pairs from Binance...")
url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
resp = requests.get(url, timeout=30)
data = resp.json()

pairs = []
for s in data['symbols']:
    if s['quoteAsset'] == 'USDT' and s['contractType'] == 'PERPETUAL':
        pairs.append(s['symbol'])

pairs = sorted(list(set(pairs)))
print(f"Total pairs: {len(pairs)}")

# Calculate date range (last 365 days)
end_time = int(time.time() * 1000)
start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

total_downloaded = 0
errors = 0

print(f"\nDownloading {len(pairs)} pairs...")
print(f"Date range: {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
print("=" * 60)

for i, symbol in enumerate(pairs):
    try:
        # Format symbol for filename
        filename = symbol.replace('USDT', '_USDT').replace('/', '_').replace(':', '_') + '-5m-futures.feather'
        filepath = os.path.join(base_dir, filename)
        
        # Download data in chunks
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&startTime={current_start}&endTime={end_time}&limit=1500"
            resp = requests.get(url, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    break
                
                for candle in data:
                    all_data.append({
                        'open_time': candle[0],
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'close_time': candle[6]
                    })
                
                current_start = data[-1][6] + 1
            else:
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Save to feather
            df.to_feather(filepath)
            total_downloaded += 1
            
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(pairs)}] Downloaded: {total_downloaded} files")
        
        # Rate limiting
        time.sleep(0.1)
        
    except Exception as e:
        errors += 1
        if errors % 10 == 0:
            print(f"Error at pair {i+1}: {str(e)[:50]}")
        time.sleep(0.5)

print("=" * 60)
print(f"✅ Download complete!")
print(f"📊 Total files: {total_downloaded}")
print(f"❌ Errors: {errors}")
print(f"📁 Location: {base_dir}")
