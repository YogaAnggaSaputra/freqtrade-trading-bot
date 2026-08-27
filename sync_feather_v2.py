#!/usr/bin/env python3
"""Sync feather files to PostgreSQL - fixed"""
import pandas as pd
import pyarrow.feather as feather
import os
from datetime import datetime
import asyncio
import asyncpg
import time

async def sync():
    # Get password from environment
    db_password = os.getenv("DB_PASSWORD", "")
    if not db_password:
        db_password = os.getenv("DB_PASSWORD", "")
    
    print(f"DB_PASSWORD length: {len(db_password)}")
    
    # PostgreSQL connection
    conn = await asyncpg.connect(
        host="postgres",
        database="botbinance",
        user="botbinance",
        password=db_password
    )
    
    # Data directory
    base_dir = '/freqtrade/user_data/data/futures'
    
    # Find all 5m feather files
    print("Scanning for feather files...")
    files = []
    for f in os.listdir(base_dir):
        if '-5m-futures.feather' in f and '_USDT_USDT' not in f:
            files.append(f)
    
    print(f"Found {len(files)} files")
    
    total = 0
    errors = 0
    
    for i, filename in enumerate(files):
        try:
            filepath = os.path.join(base_dir, filename)
            
            # Extract pair name
            pair = filename.replace('-5m-futures.feather', '').replace('_USDT', '').upper()
            
            # Read feather file
            df = feather.read_table(filepath).to_pandas()
            
            if len(df) == 0:
                continue
            
            # Check if pair exists
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM market_candles WHERE pair = $1 AND timeframe = '5m'",
                pair
            )
            
            if count > 0:
                print(f"[{i+1}/{len(files)}] Skipped {pair} (exists: {count})")
                continue
            
            # Insert data
            inserted = 0
            for _, row in df.iterrows():
                try:
                    await conn.execute("""
                        INSERT INTO market_candles (pair, timeframe, timestamp, open, high, low, close, volume, source, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (pair, timeframe, timestamp) DO NOTHING
                    """,
                        pair, "5m", row.name, float(row['open']), float(row['high']),
                        float(row['low']), float(row['close']), float(row['volume']),
                        "binance", datetime.utcnow()
                    )
                    inserted += 1
                except Exception as e:
                    pass
            
            total += inserted
            print(f"[{i+1}/{len(files)}] Inserted {inserted} candles for {pair}")
            
            time.sleep(0.05)
            
        except Exception as e:
            errors += 1
            print(f"Error with {filename}: {e}")
            time.sleep(0.1)
    
    # Verify
    result = await conn.fetchval("SELECT COUNT(*) FROM market_candles")
    pairs = await conn.fetchval("SELECT COUNT(DISTINCT pair) FROM market_candles")
    
    print(f"\n{'='*60}")
    print(f"✅ Sync Complete!")
    print(f"📊 Total new candles: {total}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Total records in DB: {result}")
    print(f"📈 Unique pairs: {pairs}")
    
    await conn.close()

asyncio.run(sync())
