import pytest
import inspect

def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.anyio)

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture(autouse=True)
def cleanup_memory():
    yield
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()

