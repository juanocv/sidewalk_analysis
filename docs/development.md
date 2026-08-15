# Development

## Local Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional ML dependencies only in environments prepared for the matching PyTorch/CUDA versions.

## Checks

Run these before opening a pull request:

```powershell
python -m compileall sidewalk_ai -q
python -m pytest
python -m ruff check sidewalk_ai
python -m black --check sidewalk_ai
```

On Windows, the same baseline checks can be run with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
```

CI runs the identical sequence on Linux for Python 3.10 and 3.13; see
`.github/workflows/checks.yml`. A pull request that leaves any of them red will fail.

## Testing Policy

- Unit tests must not require `GOOGLE_API_KEY`.
- Unit tests must not write outside `tmp_path` or the repository workspace.
- GPU, model-download, and live API checks must carry the `gpu` or `network` marker.
  `pyproject.toml` deselects both by default, so marking a test is what keeps it out of
  the default suite — run them with `pytest -m gpu` or `pytest -m network`.
- Geometry and refinement changes should include small synthetic masks/depth maps where possible.
- Estimation code must stay deterministic: thread a `seed` through instead of calling
  `np.random.*` directly, and prefer `hashlib` over `hash()` for anything that reaches output.

## Failures Must Be Audible

Every bug that survived the refactor and reached a user did so by failing quietly.
The pattern is always the same: a broad guard turns a real failure into a plausible
looking result, and the symptom appears far away from the cause.

- **Never write `except Exception: pass`.** Log at minimum. If the caller can carry
  on without whatever failed, say what was lost and why continuing is safe:
  `logger.debug("Could not outline obstacle %r: %s", label, exc)`.
- **Do not guard imports of this package's own modules.** A fallback default there
  hides a broken install and can silently change results — `core.pipeline` fell back
  to a logo-bar height of 20 px, which feeds `bottom_ignore_px` in `compute_width`.
- **Check what a tolerant load actually loaded.** `load_state_dict(..., strict=False)`
  accepts a checkpoint that matches nothing and leaves the network random. Assert the
  parameters landed; both ZoeDepth and DeepLab shipped this bug, and each surfaced
  only as an implausible measurement much later.
- **Put the reason in the message, not in a payload.** `debug_event` payloads render
  only under `SWAI_LOG_FORMAT=json`, so a failure logged that way is invisible in a
  normal run. Use `logger.warning("...: %s", exc)` for anything a user should act on.
- **Library code logs; only the CLI prints.** `print()` bypasses `--log-file` and
  `--log-format`, so it vanishes exactly when someone is capturing output to debug.
  `cli/play.py` and `diagnostics.py` print because their output *is* the product;
  everywhere else use the module logger. `cli/_debug_viz.py` used to print, and its
  "failed to write" messages never reached a log file.
- **A degraded result needs a distinct signal.** "No obstacles found" and "obstacle
  detection failed" both produce zero clearances, which the accessibility metrics
  score as a perfect rating. Make the difference visible to the caller.

Two silent handlers are deliberate and documented in place: `_fmt_num` returning
`"N/A"`, and `diagnostics` recording the exception into the report it emits. Anything
else that swallows an exception should be treated as a defect.

An AST sweep is quicker than grep for auditing this, since it can tell a handler that
logs from one that only re-binds a default.

## Git Hygiene

- Commit source, tests, docs, and configuration together when they support the same change.
- Do not commit local `.env`, caches, generated debug sheets, or root-level experiment images.
- Keep third-party checkouts as external dependencies rather than editing them as part of package changes.
