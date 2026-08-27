#!/usr/bin/env python3
import requests
import pandas as pd
import pyarrow.feather as feather
import os

# Create directory
base_dir = "/freqtrade/user_data/data/binanceusdm/futures/5m"
os.makedirs(base_dir, exist_ok=True)

# Top pairs to download
pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

print("Downloading OHLCV data...")

for pair in pairs:
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&limit=1500"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create dataframe
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'ntrades', 'tbba', 'tbqb', 'unused'
            ])
            
            # Keep only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Save
            filename = f"{base_dir}/{pair}_5m.feather"
            feather.write_feather(df, filename)
            
            print(f"✅ {pair}: {len(df)} candles")
        else:
            print(f"❌ {pair}: Error {response.status_code}")
            
    except Exception as e:
        print(f"❌ {pair}: {e}")

print("\nDownload complete!")
print(f"Total files: {len(os.listdir(base_dir))}")
