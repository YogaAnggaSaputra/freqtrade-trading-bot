import asyncio
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Callable, Any

def round_decimal(val: Any, decimals: int = 8) -> Decimal:
    """Helper to round decimal values precisely."""
    if val is None:
        return Decimal("0")
    d = Decimal(str(val))
    return d.quantize(Decimal("1." + "0" * decimals), rounding=ROUND_HALF_UP)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

async def retry_async(
    func: Callable[..., Any],
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """Retry function for async network calls."""
    current_delay = delay
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(current_delay)
            current_delay *= backoff
