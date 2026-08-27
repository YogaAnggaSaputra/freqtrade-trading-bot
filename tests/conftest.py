from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_redis():
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    return mock

@pytest.fixture
def mock_message_bus():
    mock = MagicMock()
    mock.publish = AsyncMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock
