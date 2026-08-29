# PYTHON_DEVELOPMENT.md

Day-to-day engineering idioms for this repo. Python ≥3.12, uv-managed.

## Environment

```bash
uv sync                # create/refresh .venv from pyproject.toml
uv add <pkg>           # add a runtime dep
uv add --dev <pkg>     # add a dev dep
```

`VIRTUAL_ENV` from other environments leaks into shells — if `uv` complains
about a mismatched venv, `unset VIRTUAL_ENV` first.

## Verification gate (run before every commit)

```bash
uv sync
uv run pytest -q
uvx ruff check src/ tests/
```

- `uvx ruff` is the safe lint invocation when ruff is missing from the venv;
  this repo declares ruff in `[dependency-groups].dev`, so `uv run ruff`
  also works — use whichever resolves.
- NEVER chain `pytest | tail && git commit`: the pipe masks pytest's exit
  code and commits broken state. Run the gate, inspect the exit code, then
  commit.

## Typing & models

- Pydantic v2: models for every schema (Trace, Action, Outcome, Scenario,
  KnowledgeRecord, Patch). `model_validate_json` / `model_dump_json` at IO
  boundaries. Set `extra` explicitly (`forbid` for contracts).
- `datetime.now(timezone.utc)`; never `datetime.utcnow()` (deprecated).
- `list[X] | None`, `dict[str, Y]` — no `Optional`, no `typing.List`.
- Ruff rules E F I UP B SIM; line-length 100. `uv run ruff format` before
  commit.

## Typer CLI

- `src/cli.py` is a thin dispatch layer; commands import from subpackages.
- Test commands with `typer.testing.CliRunner` (see tests/test_cli_smoke.py).
- **Typer 0.27 single-command collapse:** `get_command(app)` returns the lone
  command itself (no group) when the app has exactly one registered command,
  so `CliRunner().invoke(app, ["version"])` fails with "unexpected extra
  arguments" and flips behavior as soon as a second command is added. Keep a
  root `@app.callback()` in `cli.py` so the app is always a Group.
- Keep business logic out of commands — call functions so unit tests don't
  need the CLI runner.

## Determinism

- `random.Random(seed)` per run, never the global RNG.
- World reset per task; no module-level mutable state in `world/`.
- Every experiment writes a manifest (SPEC.md §19).

## LLM integration

- All calls go through `agent`'s adapter. Capture tokens + estimated cost per
  call into the trace metrics — cost evidence lives in the store.
- Unit tests use a `FakeModel` (scripted responses); live calls are env-gated
  (`EXP_LLM_API_KEY` set) integration tests that assert structure, not quality.

## Pitfalls (proven elsewhere, applies here)

- `datetime` naive comparisons break across runs — always UTC-aware.
- JSONL writes must flush/close before a reader opens the file (no
  concurrent read/write in-process is fine, but tests must use `with`).
- YAML `yaml.safe_load` everywhere; never `yaml.load`.
