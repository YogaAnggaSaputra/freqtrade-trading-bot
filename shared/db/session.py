from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import text
from shared.db.models import Base
from shared.security import get_secret
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}".format(
        user=os.getenv("DB_USER", "botbinance"),
        password=get_secret("db_password") or os.getenv("DB_PASSWORD", "changeme"),
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        db=os.getenv("DB_NAME", "botbinance"),
    )
)

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Fixed advisory lock id so concurrent init_db() callers serialize schema creation.
# Postgres CREATE TYPE is not idempotent — multiple services racing create_all()
# will hit UniqueViolationError on enum types like ordersideenum.
_SCHEMA_LOCK_ID = 84210001


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:id)"), {"id": _SCHEMA_LOCK_ID})
        try:
            await conn.run_sync(Base.metadata.create_all)
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": _SCHEMA_LOCK_ID})


async def close_db():
    await engine.dispose()