import os
from shared.security import get_secret

class Config:
    def __init__(self):
        self.db_user = os.getenv("DB_USER", "botbinance")
        self.db_password = get_secret("db_password") or os.getenv("DB_PASSWORD", "changeme")
        self.db_host = os.getenv("DB_HOST", "postgres")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "botbinance")
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = get_secret("redis_password") or os.getenv("REDIS_PASSWORD")
        self.trade_mode = os.getenv("TRADE_MODE", "demo")

config = Config()
