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

The base install is enough for unit tests and lightweight package imports. The examples in
the **Running** section load real segmentation/depth models, so install the ML layer before
running them:

```powershell
python -m pip install -e ".[ml]"
```

On Windows, install the PyTorch build that matches your CPU/CUDA setup and install
backend-specific packages such as Detectron2, OneFormer, or ZoeDepth according to their upstream
instructions when you select those backends.

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` before running Street View calls.
The key is required only when the code performs an actual Google API request; importing modules
and running unit tests do not require it.

For backend-specific Windows/CUDA guidance, diagnostics, and logging setup, see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Running

The editable install exposes a `sidewalk-ai` console script; `python -m sidewalk_ai.cli.play`
runs the same entry point if you prefer not to rely on the installed script.

Single image:

```powershell
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg --single-view --device cpu
```

This default command uses `--seg oneformer --depth zoe`, which requires the optional ML layer and
the corresponding model assets. Add `--debug` only after installing debug/backend visualization
dependencies.

Coordinates:

```powershell
sidewalk-ai --lat -23.678479 --lon -46.559621 --multi-view --device cuda
```

Address:

```powershell
sidewalk-ai "Av. Paulista 1578, Sao Paulo" --multi-view --device cuda
```

Use `--debug --outdir debug_out` to write diagnostic images.

Pipeline knobs worth knowing:

- `--seg a+b+c` runs an ensemble; `--ensemble-method or|and|majority` picks the fusion rule.
  Only the sidewalk mask is fused — obstacles come from the panoptic map of the first member
  that provides one, intersected with the fused mask. `majority` needs three or more members
  to differ from `and`.
- `--no-refine` feeds the raw segmenter mask downstream instead of the refined one.
- `--fallback-scale` / `--force-fallback` control how a *relative* depth back-end
  (MiDaS) is converted to metres. They have no effect with `--depth zoe`, which is
  already metric. See [`docs/reproducibility.md`](docs/reproducibility.md#metric-scale-for-relative-depth-back-ends).

## Web API

The same pipeline is exposed over HTTP for integration with external systems:

```powershell
python -m pip install -e ".[api,ml]"
uvicorn sidewalk_ai.webapi:app --host 127.0.0.1 --port 8000
```

`POST /analyse/single` and `POST /analyse/multi` return width, clearances and the
NBR 9050 accessibility rating; `GET /ping` is a liveness probe. Interactive schema
docs are served at `/docs`.

The API has no authentication and calls a paid Google API on every request — keep
it behind a proxy or bound to localhost. See [`docs/webapi.md`](docs/webapi.md)
for endpoints, configuration, and the concurrency model.

## Diagnostics

Runtime diagnostics:

```powershell
python -m sidewalk_ai.diagnostics
python -m sidewalk_ai.diagnostics --json
```

Structured logs:

```powershell
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg `
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

These same four checks run in CI on Linux for Python 3.11 and 3.13
(`.github/workflows/checks.yml`).

The default suite is unit-level: it never downloads models, calls Google APIs, or needs a GPU.
That is enforced, not just documented — `pyproject.toml` deselects the `gpu` and `network`
markers by default. Run the excluded checks deliberately:

```powershell
python -m pytest -m gpu
python -m pytest -m network
```

Add new heavyweight checks with one of those markers so they stay out of the default run.

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
