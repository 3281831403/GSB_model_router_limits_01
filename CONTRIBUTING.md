# Contributing

Thanks for helping improve model-router-python! Bug reports, docs fixes, new tests and features are all welcome.

## Set up

```bash
git clone https://github.com/TheCoder30ec4/model_router_python.git
cd model_router_python
uv sync --group dev          # creates .venv with pytest and ruff
```

No `uv`? `python -m venv .venv && .venv/bin/pip install -e . pytest ruff` works too.

## Before you open a pull request

```bash
uv run pytest                # all tests are offline; no API key needed
uv run ruff check .
uv run ruff format .
```

- **Tests must stay offline.** Mock network calls with `unittest.mock.patch` (see `tests/`), so CI never needs real keys.
- **No runtime dependencies.** The library uses only the standard library. Please discuss in an issue first before adding one.
- **Keep changes focused.** One fix or feature per pull request, with a test that fails without it.
- **Never commit keys.** `.env` is git-ignored; keep it that way.

## Trying it for real

The examples call live APIs and cost a few cents:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
uv run python examples/basic.py
```

## Project layout

```
src/model_router/
  router.py    Router: candidate selection, limit filtering, route()
  catalog.py   live model list from OpenRouter (cached)
  jev.py       the Jev routing request (direct or via OpenRouter)
  models.py    ModelInfo, Limits dataclasses
  errors.py    exception types
  _http.py     tiny urllib JSON helper
tests/         offline unit tests, one file per module
examples/      runnable demos (need real keys)
```

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and add an entry to `CHANGELOG.md`.
2. Merge to `main`.
3. Create a GitHub Release tagged `v<version>`. `.github/workflows/deployment.yml` tests, builds and publishes to PyPI.

By contributing you agree your work is released under the [MIT License](LICENSE) and that you'll follow the [Code of Conduct](CODE_OF_CONDUCT.md).
