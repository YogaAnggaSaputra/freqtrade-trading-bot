#!/usr/bin/env python3
"""Download OHLCV for all available pairs from feather files and sync to PostgreSQL"""
import pandas as pd
import pyarrow.feather as feather
import os
from datetime import datetime
import asyncio
from sqlalchemy import create_engine, text
import time

# PostgreSQL connection
DB_URL = "postgresql://botbinance:Changeme123@postgres:5432/botbinance"
engine = create_engine(DB_URL)

# Data directory
base_dir = '/freqtrade/user_data/data/futures'

# Find all 5m feather files
print("Scanning for feather files...")
files = []
for f in os.listdir(base_dir):
    if '-5m-futures.feather' in f and '_USDT_USDT' not in f:
        files.append(f)

print(f"Found {len(files)} files")

# Convert each file to PostgreSQL
total = 0
errors = 0

for i, filename in enumerate(files[:50]):  # First 50 pairs
    try:
        filepath = os.path.join(base_dir, filename)
        
        # Extract pair name
        pair = filename.replace('-5m-futures.feather', '').replace('_USDT', '').upper()
        
        # Read feather file
        df = feather.read_table(filepath).to_pandas()
        
        if len(df) == 0:
            continue
        
        # Prepare for insertion
        df['pair'] = pair
        df['timeframe'] = '5m'
        df['source'] = 'binance'
        df['created_at'] = datetime.utcnow()
        
        # Insert to PostgreSQL
        with engine.begin() as conn:
            # Check if pair exists
            result = conn.execute(text("SELECT COUNT(*) FROM market_candles WHERE pair = :pair AND timeframe = '5m'"), {"pair": pair})
            count = result.scalar()
            
            if count == 0:
                # Insert new data
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO market_candles (pair, timeframe, timestamp, open, high, low, close, volume, source, created_at)
                        VALUES (:pair, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :source, :created_at)
                        ON CONFLICT (pair, timeframe, timestamp) DO NOTHING
                    """), {
                        "pair": pair,
                        "timeframe": "5m",
                        "timestamp": row.name,
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": float(row['volume']),
                        "source": "binance",
                        "created_at": datetime.utcnow()
                    })
                total += len(df)
                print(f"[{i+1}/{len(files[:50])}] Inserted {len(df)} candles for {pair}")
            else:
                print(f"[{i+1}/{len(files[:50])}] Skipped {pair} (already exists: {count} records)")
        
        time.sleep(0.1)
        
    except Exception as e:
        errors += 1
        print(f"Error with {filename}: {e}")
        time.sleep(0.5)

print(f"\n{'='*60}")
print(f"✅ Complete!")
print(f"📊 Total candles inserted: {total}")
print(f"❌ Errors: {errors}")

# Verify
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM market_candles"))
    print(f"📁 Total records in DB: {result.scalar()}")
    
    result = conn.execute(text("SELECT COUNT(DISTINCT pair) FROM market_candles"))
    print(f"📈 Unique pairs: {result.scalar()}")
