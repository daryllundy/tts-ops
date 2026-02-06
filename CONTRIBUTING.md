# Contributing

## Branching strategy

- `main`: production-ready, stable code only.
- `develop`: integration branch for approved features and fixes.
- `codex/*` or `feature/*`: short-lived work branches merged into `develop`.

## Recommended workflow

1. Branch from `develop`.
2. Keep changes focused and small.
3. Run checks locally before opening a PR:
   - `ruff check .`
   - `mypy src`
   - `pytest -v`
4. Open a PR into `develop`.
5. After validation and review, promote `develop` to `main` as needed.

## Commit conventions

- Use clear, imperative commit messages.
- Prefer a conventional style when possible, for example:
  - `feat: add streaming fallback`
  - `fix: handle websocket disconnect`
  - `docs: update deployment runbook`

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install
```
