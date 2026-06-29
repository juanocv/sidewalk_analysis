# Sidewalk AI

Sidewalk AI estimates sidewalk width and obstacle clearance from Google Street View imagery.
The production package lives in `sidewalk_ai/`; research notebooks, experiments, image samples,
and third-party model checkouts are kept outside the package boundary.

## What It Does

The pipeline combines:

- Street View image acquisition and geocoding.
- Sidewalk segmentation through Detectron2, OneFormer, DeepLab, or ensemble backends.
- Monocular depth estimation through MiDaS or ZoeDepth.
- Mask refinement, width estimation, obstacle extraction, and accessibility metrics.
- Single-view and multi-view analysis for address or coordinate inputs.

## Repository Layout

```text
sidewalk_ai/              Python package used by the CLI and API
sidewalk_ai/core/         Pipeline orchestration and shared configuration
sidewalk_ai/io/           Street View, image, and geospatial I/O
sidewalk_ai/models/       Model adapters and factories
sidewalk_ai/processing/   Geometry, fusion, refinement, and accessibility logic
sidewalk_ai/tests/        Unit tests that avoid network and GPU dependencies
prototype/                Research/prototype code kept out of the package build
generic/                  Local datasets, notebooks, and experiment outputs
```

Large third-party repositories such as Detectron2, OneFormer, ZoeDepth, and DeepLab are treated
as local external dependencies and are excluded from the Python package build.

## Setup

Use Python 3.11 or newer. GPU model stacks may require a stricter Python/PyTorch/CUDA matrix, so
install those backends according to their upstream documentation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the full ML stack, install the backend-specific packages after the base setup:

```powershell
python -m pip install -e ".[ml]"
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` before running Street View calls.
The key is required only when the code performs an actual Google API request; importing modules
and running unit tests do not require it.

For backend-specific Windows/CUDA guidance, diagnostics, and logging setup, see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Running

Single image:

```powershell
python -m sidewalk_ai.cli.play --image generic/images/streetview_id1_heading0.jpg --single-view --device cpu
```

Coordinates:

```powershell
python -m sidewalk_ai.cli.play --lat -23.678479 --lon -46.559621 --multi-view --device cuda
```

Address:

```powershell
python -m sidewalk_ai.cli.play "Av. Paulista 1578, Sao Paulo" --multi-view --device cuda
```

Use `--debug --outdir debug_out` to write diagnostic images.

Runtime diagnostics:

```powershell
python -m sidewalk_ai.diagnostics
python -m sidewalk_ai.diagnostics --json
```

Structured logs:

```powershell
python -m sidewalk_ai.cli.play --image generic/images/streetview_id1_heading0.jpg `
  --single-view --device cpu --log-level DEBUG --log-format json --log-file debug_out/run.jsonl
```

## Quality Checks

```powershell
python -m compileall sidewalk_ai -q
python -m pytest
python -m ruff check sidewalk_ai
python -m black --check sidewalk_ai
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
```

The default tests are unit-level and should not download models, call Google APIs, or require a GPU.
Heavy model checks should be added as explicit integration tests with `gpu` or `network` markers.

## Version-Control Hygiene

- Keep generated PNG/PDF/HTML outputs out of commits unless they are intentional documentation assets.
- Keep secrets in `.env`; commit only `.env.example`.
- Keep third-party model repositories outside the package build.
- Prefer focused tests around geometry, request normalization, and pipeline orchestration before
  changing model adapters or estimation heuristics.

## Citation

```bibtex
@misc{sidewalk_ai_2025,
  author = {Diego Guerra and Juan Oliveira de Carvalho},
  title = {Automatic Sidewalk Width Estimation and Obstacle Detection Using Panoptic Segmentation and Depth Estimation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/juanocv/sidewalk-analysis}
}
```
