#!/usr/bin/env python3
"""Download OHLCV data for all 654 Binance Futures pairs"""
import subprocess
import time
import os
import requests

# Get all pairs from Binance API
print("Fetching pairs from Binance...")
url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
resp = requests.get(url, timeout=30)
data = resp.json()

pairs = []
for s in data['symbols']:
    if s['quoteAsset'] == 'USDT' and s['contractType'] == 'PERPETUAL':
        pairs.append(s['symbol'].replace('USDT:USDT', 'USDT'))

pairs = sorted(list(set(pairs)))
print(f"Total USDT futures pairs: {len(pairs)}")

# Save pair list
with open('/tmp/pairs_list.txt', 'w') as f:
    f.write('\n'.join(pairs))

# Download in batches using freqtrade CLI
batch_size = 20
total_downloaded = 0
failed = 0

print(f"\nStarting download with batches of {batch_size} pairs...")
print(f"Timerange: 2024-01-01 to 2025-12-31")
print("=" * 60)

for i in range(0, len(pairs), batch_size):
    batch = pairs[i:i+batch_size]
    pair_str = ' '.join([f'{p}/USDT:USDT' for p in batch])
    
    cmd = [
        'python3', '-m', 'freqtrade', 'download-data',
        '--exchange', 'binanceusdm',
        '--pairs', pair_str,
        '--timeframes', '5m',
        '--timerange', '20240101-20251231',
        '--trading-mode', 'futures',
        '--data-format-ohlcv', 'feather',
        '--datadir', '/freqtrade/user_data/data'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Count successful downloads
        count = result.stdout.count('Downloaded data')
        total_downloaded += count
        
        # Progress every 100 pairs
        if i % 100 == 0 or i + batch_size >= len(pairs):
            progress = min(i + batch_size, len(pairs))
            print(f"[{progress}/{len(pairs)}] Files downloaded: {total_downloaded}")
        
        # Small delay between batches
        time.sleep(0.3)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout at pair {i}")
        failed += 1
    except Exception as e:
        print(f"Error at pair {i}: {e}")
        failed += 1

print("=" * 60)
print(f"✅ Download complete!")
print(f"📊 Total files: {total_downloaded}")
print(f"❌ Failed batches: {failed}")
print(f"📁 Data location: /freqtrade/user_data/data/")
