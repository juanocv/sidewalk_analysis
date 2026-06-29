$ErrorActionPreference = "Stop"

python -m compileall sidewalk_ai -q
python -m pytest
python -m ruff check sidewalk_ai
python -m black --check sidewalk_ai
python -m sidewalk_ai.diagnostics
