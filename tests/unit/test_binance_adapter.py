import os
import sys
from decimal import Decimal

import pytest

_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "binance-adapter",
)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from adapter import BinanceFuturesAdapter

from shared.schemas import OrderStatus


class TestBinanceAdapter:
    @pytest.mark.asyncio
    async def test_parse_ticker(self):
        adapter = BinanceFuturesAdapter()
        raw_payload = {
            "s": "BTCUSDT",
            "c": "50000.0",
            "b": "49999.0",
            "a": "50001.0",
            "B": "1.5",
            "A": "2.0",
            "E": 1700000000000,
        }
        res = adapter._parse_ticker(raw_payload)
        assert res is not None
        assert res.pair == "BTCUSDT"
        assert res.last_price == Decimal("50000.0")
        assert res.spread == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_parse_candle(self):
        adapter = BinanceFuturesAdapter()
        raw_payload = {
            "k": {
                "s": "BTCUSDT",
                "i": "5m",
                "t": 1700000000000,
                "o": "50000",
                "h": "50100",
                "l": "49900",
                "c": "50050",
                "v": "10.5",
            }
        }
        res = adapter._parse_candle(raw_payload)
        assert res is not None
        assert res.pair == "BTCUSDT"
        assert res.close == Decimal("50050")
        assert res.volume == Decimal("10.5")

    @pytest.mark.asyncio
    async def test_parse_order_update(self):
        adapter = BinanceFuturesAdapter()
        raw_payload = {
            "o": {
                "s": "BTCUSDT",
                "i": "12345",
                "c": "my_client_id",
                "S": "BUY",
                "o": "LIMIT",
                "X": "FILLED",
                "q": "0.5",
                "z": "0.5",
                "p": "50000",
                "ap": "50000",
                "T": 1700000000000,
            }
        }
        res = adapter._parse_order_update(raw_payload)
        assert res is not None
        assert res.order_id == "12345"
        assert res.client_order_id == "my_client_id"
        assert res.status == OrderStatus.FILLED
        assert res.amount == Decimal("0.5")
        assert res.avg_price == Decimal("50000")
