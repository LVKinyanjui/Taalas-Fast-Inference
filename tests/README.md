# Test Suite

Comprehensive test suite for the ChatJimmy OpenAI-Compatible API adapter.

## Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── test_utils.py         # Unit tests for utility functions
├── test_endpoints.py     # Integration tests for API endpoints
├── test_errors.py        # Tests for error handling and auth
├── test_upstream.py      # Tests for upstream ChatJimmy integration
└── test_streaming.py     # Tests for streaming and edge cases
```

## Running Tests

### Install test dependencies

```bash
# Using uv (recommended)
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

### Run all tests

```bash
# Using uv
uv run pytest

# Or directly (with pythonpath)
python -m pytest
```

### Run specific test file

```bash
pytest tests/test_endpoints.py -v
```

### Run with coverage

```bash
uv run pytest --cov=server --cov-report=term-missing
```

### Run only unit tests (fast)

```bash
pytest tests/test_utils.py tests/test_upstream.py -v
```

### Run only integration tests

```bash
pytest tests/test_endpoints.py tests/test_errors.py tests/test_streaming.py -v
```

## Test Categories

| File | Type | Coverage |
|------|------|----------|
| `test_utils.py` | Unit | Message conversion, parameter mapping, usage building |
| `test_upstream.py` | Unit | `call_chatjimmy` function with mocked HTTP |
| `test_endpoints.py` | Integration | All API endpoints with mocked upstream |
| `test_errors.py` | Integration | Error responses, auth, format compliance |
| `test_streaming.py` | Integration | SSE format, edge cases, validation |

## Fixtures

Defined in `conftest.py`:

- `client` - FastAPI TestClient
- `mock_chatjimmy_response` - Sample upstream response with stats
- `mock_chatjimmy_response_no_stats` - Response without stats
- `sample_messages` - Typical conversation messages
- `sample_tools` - Tool definitions for function calling
- `mock_upstream_call` - Auto-mocks `call_chatjimmy` for all tests

## Mocking Strategy

- **Unit tests**: Mock `requests.post` directly
- **Integration tests**: Use `mock_upstream_call` fixture which patches `server.call_chatjimmy`
- **Auth tests**: Temporarily set `CHATJIMMY_API_KEY` env var and reload module

## Continuous Integration

Add to your CI pipeline:

```yaml
- name: Run tests
  run: |
    uv sync --extra dev
    uv run pytest --cov=server --cov-report=xml
```

## Writing New Tests

1. Add test file in `tests/` with `test_` prefix
2. Use existing fixtures from `conftest.py`
3. Follow naming convention: `Test<Feature>` class with `test_<scenario>` methods
4. Mock upstream calls - never hit real ChatJimmy API in tests