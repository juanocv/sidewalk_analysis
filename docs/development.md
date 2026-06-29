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

## Testing Policy

- Unit tests must not require `GOOGLE_API_KEY`.
- Unit tests must not write outside `tmp_path` or the repository workspace.
- GPU, model-download, and live API checks should be marked and kept out of the default suite.
- Geometry and refinement changes should include small synthetic masks/depth maps where possible.

## Git Hygiene

- Commit source, tests, docs, and configuration together when they support the same change.
- Do not commit local `.env`, caches, generated debug sheets, or root-level experiment images.
- Keep third-party checkouts as external dependencies rather than editing them as part of package changes.
