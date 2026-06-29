# Reproducibility

This project should be installed in layers. The base package and unit tests must work without
GPU, model downloads, or Google API access. Vision backends are optional and should be validated
one at a time.

## Supported Baseline

- Python 3.11 or newer.
- Windows, Linux, or macOS for the base package.
- CPU-only mode for tests and diagnostics.
- CUDA GPU only when using heavy model backends that require it.

The ML stack is intentionally not pinned to one CUDA build in `requirements.txt`, because PyTorch,
Detectron2, OneFormer, and ZoeDepth have OS/GPU-specific installation constraints.

## Install Layers

Base development install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1
```

API-only additions:

```powershell
python -m pip install -e ".[api]"
```

Generic ML additions:

```powershell
python -m pip install -e ".[ml]"
```

Install backend-specific packages separately when needed:

- PyTorch: install the CPU or CUDA build that matches the machine.
- Detectron2: install a build compatible with the local PyTorch/CUDA/OS matrix.
- OneFormer: install its upstream dependencies and model assets.
- ZoeDepth: install or clone the backend according to the target environment.

## Diagnostics

Run diagnostics before loading any model weights:

```powershell
python -m sidewalk_ai.diagnostics
python -m sidewalk_ai.diagnostics --json
```

After editable installation, this console script is also available:

```powershell
sidewalk-ai-diagnostics
```

The diagnostic command reports Python, OS, package versions, PyTorch/CUDA status, selected
environment variables, and import availability for optional backends.

## Logging

Logging is centralized under the `sidewalk_ai` logger.

Environment configuration:

```powershell
$env:SWAI_LOG_LEVEL = "DEBUG"
$env:SWAI_LOG_FORMAT = "json"
$env:SWAI_LOG_FILE = "debug_out/pipeline.jsonl"
```

CLI configuration:

```powershell
python -m sidewalk_ai.cli.play --image generic/images/streetview_id1_heading0.jpg `
  --single-view --device cpu --log-level DEBUG --log-format json --log-file debug_out/run.jsonl
```

Use `--debug` only when image/debug artifacts are needed. Use `--log-level DEBUG` when textual
diagnostics are enough.

## Model Cache Hygiene

Use explicit cache directories for reproducible machines and CI runners:

```powershell
$env:TORCH_HOME = "C:\models\torch"
$env:HF_HOME = "C:\models\huggingface"
```

Document the cache location and model variants used in experiments. Do not commit downloaded
weights, generated debug images, or local API keys.

## Minimum Validation

Before debugging model-specific issues, confirm the base package:

```powershell
python -m compileall sidewalk_ai -q
python -m pytest
python -m ruff check sidewalk_ai
python -m black --check sidewalk_ai
python -m sidewalk_ai.diagnostics
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
```
