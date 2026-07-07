# Tests Directory

This directory is reserved for future unit tests and integration tests.

Currently empty - test files were removed as they were demo scripts, not actual tests.

## To Add Tests

Create pytest-compatible test files:

```python
# test_example.py
import pytest
from retrieval_playground.utils.model_manager import ModelManager

def test_model_manager_singleton():
    """Test that ModelManager is a singleton."""
    manager1 = ModelManager()
    manager2 = ModelManager()
    assert manager1 is manager2
```

Run with: `pytest retrieval_playground/tests/`
