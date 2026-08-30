import asyncio
import os
import asyncpg

# Database connection parameters from environment
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "botbinance")
DB_USER = os.getenv("DB_USER", "botbinance")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

# Connection pool
_pool = None

async def get_pool():
    global _pool
    if _pool is None:
        conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        _pool = await asyncpg.create_pool(dsn=conn_string)
    return _pool

async def get_candles_async(pair: str, timeframe: str, limit: int = 10):
    """Async function to get candles from PostgreSQL"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        sql = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_candles
            WHERE pair = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await conn.fetch(sql, pair, timeframe, limit)
        return [
            {
                "timestamp": str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for row in rows
        ]
