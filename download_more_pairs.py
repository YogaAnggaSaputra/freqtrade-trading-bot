#!/usr/bin/env python3
"""Download OHLCV for more pairs with proper rate limiting"""
import subprocess
import time
import requests
import json
from datetime import datetime, timedelta

# Get pairs from existing files
pairs_done = set()
import os
base_dir = '/freqtrade/user_data/data/futures'
for f in os.listdir(base_dir):
    if '-5m-futures.feather' in f:
        pairs_done.add(f.replace('-5m-futures.feather', '').replace('_USDT', ''))

print(f"Already have data for {len(pairs_done)} pairs")

# Get all pairs from Binance
print("Fetching all pairs from Binance...")
url = 'https://fapi.binance.com/fapi/v2/ticker/price'
resp = requests.get(url, timeout=30)
prices = resp.json()

all_pairs = []
for p in prices:
    symbol = p['symbol']
    if symbol.endswith('USDT') and '/USDT' not in symbol:
        pair = symbol.replace('USDT', '')
        all_pairs.append(pair)

all_pairs = sorted(list(set(all_pairs)))
print(f"Total available: {len(all_pairs)} pairs")

# Filter out pairs we already have
to_download = [p for p in all_pairs if p not in pairs_done]
print(f"Pairs to download: {len(to_download)}")

# Download in small batches with longer delay
batch_size = 10
delay = 2.0  # seconds between batches

total_downloaded = 0
failed = 0

print(f"\nStarting download (batch size: {batch_size}, delay: {delay}s)...")
print(f"Date range: 2024-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 60)

for i in range(0, min(len(to_download), 200), batch_size):  # Max 200 new pairs
    batch = to_download[i:i+batch_size]
    pair_str = ' '.join([f'{p}/USDT:USDT' for p in batch])
    
    cmd = [
        'python3', '-m', 'freqtrade', 'download-data',
        '--exchange', 'binanceusdm',
        '--pairs', pair_str,
        '--timeframes', '5m',
        '--timerange', '20240101-' + datetime.now().strftime('%Y%m%d'),
        '--trading-mode', 'futures',
        '--data-format-ohlcv', 'feather',
        '--datadir', base_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        # Count successful downloads
        count = result.stdout.count('Downloaded data')
        total_downloaded += count
        
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(to_download):
            progress = min(i + batch_size, len(to_download))
            print(f"[{progress}/{len(to_download)}] Downloaded: {total_downloaded} files")
        
        time.sleep(delay)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout at {i}")
        failed += 1
    except Exception as e:
        print(f"Error at {i}: {e}")
        failed += 1

print("=" * 60)
print(f"✅ Complete!")
print(f"📊 New files: {total_downloaded}")
print(f"❌ Errors: {failed}")

# Save list of downloaded pairs
with open('/tmp/downloaded_pairs.txt', 'w') as f:
    f.write('\n'.join(to_download[:total_downloaded]))
