

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "asyncio: mark a test as an asyncio test")
