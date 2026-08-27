#!/usr/bin/env python3
"""Download OHLCV data for first 100 Binance Futures pairs"""
import requests
import pandas as pd
import pyarrow.feather as feather
import time
import os
from datetime import datetime

base_dir = '/freqtrade/user_data/data/futures'
os.makedirs(base_dir, exist_ok=True)

print('Fetching pairs...')
url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
resp = requests.get(url, timeout=30)
data = resp.json()

pairs = []
for s in data['symbols']:
    if s['quoteAsset'] == 'USDT' and s['contractType'] == 'PERPETUAL':
        pairs.append(s['symbol'])

pairs = sorted(list(set(pairs)))
print(f'Total: {len(pairs)} pairs')

# Correct date range
start_time = int(datetime(2024, 1, 1).timestamp() * 1000)
end_time = int(datetime.now().timestamp() * 1000)
print(f'Date range: 2024-01-01 to {datetime.now().strftime("%Y-%m-%d")}')

total = 0
for i, symbol in enumerate(pairs[:100]):  # First 100 pairs
    try:
        filename = symbol.replace('USDT', '_USDT').replace('/', '_').replace(':', '_') + '-5m-futures.feather'
        filepath = os.path.join(base_dir, filename)
        
        all_data = []
        current_start = start_time
        while current_start < end_time:
            url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&startTime={current_start}&endTime={end_time}&limit=1500'
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                d = resp.json()
                if not d:
                    break
                for c in d:
                    all_data.append({'open_time': c[0], 'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])})
                current_start = d[-1][6] + 1
            else:
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.to_feather(filepath)
            total += 1
            if (i+1) % 25 == 0:
                print(f'[{i+1}/100] Downloaded: {total} files')
        time.sleep(0.2)
    except Exception as e:
        pass

print(f'Done! Total: {total} files')
