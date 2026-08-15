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

Needs Python 3.10 or newer, on Windows, Linux or macOS — 3.10 is what Ubuntu 22.04 LTS
ships, and CI tests both ends of that range. GPU model stacks may require a stricter
Python/PyTorch/CUDA matrix, so install those backends according to their upstream
documentation.

**Linux** — Debian and Ubuntu do not ship `venv` with the interpreter, and OpenCV links
against libGL, so install those two first.

```bash
sudo apt install python3-venv libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The base install is enough for unit tests and lightweight package imports. The examples in
the **Running** section load real segmentation/depth models, so install the ML layer before
running them — the same command on both systems:

```bash
python -m pip install -e ".[ml]"
```

That layer pins `timm`, which ZoeDepth needs, but PyTorch itself is left to you: install the
CPU or CUDA build that matches your machine from
[pytorch.org](https://pytorch.org/get-started/locally/).

Which build that is comes down to one question — does the machine have an NVIDIA GPU?

```bash
nvidia-smi     # prints a table with a driver version, or "command not found"
```

No output means no usable GPU, so take the CPU build and run every example below with
`--device cpu`. Two things that look like evidence of a GPU are not: an `nvcc` on the
`PATH` is just the `nvidia-cuda-toolkit` apt package, and `libcuda.so` can be left behind
by `libnvidia-compute-*` with no hardware or kernel module under it. The CUDA wheels bundle
their own runtime anyway, so a system CUDA version never has to match the wheel you pick —
only the driver has to be new enough.

Detectron2 and the DeepLab checkout have their own steps, described in
[`docs/reproducibility.md`](docs/reproducibility.md). Detectron2 in particular compiles from
source and needs a toolchain that a stock Ubuntu does not have
(`sudo apt install build-essential python3-dev`). OneFormer and ZoeDepth need no separate
install.

A helper does the venv and the install in one step, with the optional layers behind flags:

```bash
./scripts/setup-dev.sh --api --ml                                         # Linux/macOS
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1 -WithApi -WithMl   # Windows
```

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` before running Street View calls.
The key is required only when the code performs an actual Google API request; importing modules
and running unit tests do not require it.

## Running

The editable install exposes a `sidewalk-ai` console script; `python -m sidewalk_ai.cli.play`
runs the same entry point if you prefer not to rely on the installed script.

Single image:

```bash
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg --single-view --device cpu
```

This default command uses `--seg oneformer --depth zoe`, which requires the optional ML layer and
the corresponding model assets. Add `--debug` only after installing debug/backend visualization
dependencies.

The first run downloads those assets and nothing warns you beforehand: roughly 1.7 GB for
OneFormer into `HF_HOME`, plus ~1.3 GB for ZoeDepth into `TORCH_HOME` (`--depth midas` pulls
~1.5 GB there instead). They are cached, so only the first run pays. Set both variables first
if the default `~/.cache` is not where they belong — see
[`docs/reproducibility.md`](docs/reproducibility.md#model-cache-hygiene).

Expect tens of seconds per frame on CPU, not seconds. Measured on one laptop CPU, single view:
this default command takes about 53 s end to end, and about 22 s with `--depth midas` instead.
Segmentation is the smaller half — roughly 16 s for OneFormer, against 0.6 s for DeepLab with
the MobileNet checkpoint.

Coordinates:

```bash
sidewalk-ai --lat -23.678479 --lon -46.559621 --multi-view --device cuda
```

Address:

```bash
sidewalk-ai "Av. Paulista 1578, Sao Paulo" --multi-view --device cuda
```

Those two examples name `--device cuda`, which fails loudly on a machine without a usable GPU
rather than falling back. Swap it for `--device cpu`, or drop the flag entirely: the default is
`--device auto`, which picks CUDA when the installed PyTorch can and CPU otherwise.

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

```bash
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

```bash
python -m sidewalk_ai.diagnostics
python -m sidewalk_ai.diagnostics --json
```

Structured logs:

```bash
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg \
  --single-view --log-level DEBUG --log-format json --log-file debug_out/run.jsonl
```

On PowerShell the line continuation is a backtick rather than a backslash:

```powershell
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg `
  --single-view --log-level DEBUG --log-format json --log-file debug_out/run.jsonl
```

## Quality Checks

```bash
python -m compileall sidewalk_ai -q
python -m pytest
python -m ruff check sidewalk_ai
python -m black --check sidewalk_ai
```

Or through the helper that wraps all of them, including diagnostics:

```bash
./scripts/check.sh                                              # Linux/macOS
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1     # Windows
```

These same four checks run in CI on Linux for Python 3.10 and 3.13
(`.github/workflows/checks.yml`).

The default suite is unit-level: it never downloads models, calls Google APIs, or needs a GPU.
That is enforced, not just documented — `pyproject.toml` deselects the `gpu` and `network`
markers by default. Run the excluded checks deliberately:

```bash
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
