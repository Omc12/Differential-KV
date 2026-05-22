import pytest
import inspect

def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.anyio)

@pytest.fixture
def anyio_backend():
    return 'asyncio'
