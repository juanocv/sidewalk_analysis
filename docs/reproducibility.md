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

For a new workstation intended to run the README examples, install both layers in the same
environment:

```powershell
python -m pip install -e ".[dev,ml]"
```

Install backend-specific packages separately when needed:

- PyTorch: install the CPU or CUDA build that matches the machine.
- Detectron2: install a build compatible with the local PyTorch/CUDA/OS matrix.
- OneFormer: install its upstream dependencies and model assets.
- ZoeDepth: install or clone the backend according to the target environment.

### The timm pin

`ml` pins `timm==0.6.13`, and the window is genuinely that narrow:

- ZoeDepth builds its core through `torch.hub.load("intel-isl/MiDaS", ...)`, whose
  `hubconf` imports `timm`. Without it the default `--depth zoe` cannot start at all.
- timm 1.x renamed the BEiT block internals (`drop_path` became `drop_path1`/`drop_path2`),
  which MiDaS calls directly, and made `relative_position_index` a non-persistent buffer
  that the published checkpoints still carry.
- timm 0.6.12 and older fail to import on Python 3.11+ (mutable dataclass default in
  `timm.models.maxxvit`), and this project requires 3.11.

ZoeDepth's own `environment.yml` pins 0.6.12; 0.6.13 is the first release that also runs
on the supported Python. Loosen the pin only after checking both ends on a real image.

### Known-benign warnings

Recent `transformers` releases print a load report for the OneFormer checkpoint:

```text
...swin.encoder.layers.*.attention.self.relative_position_index | UNEXPECTED
model.pixel_level_module.encoder.swin.layernorm.{weight,bias}   | MISSING
```

Both are safe to ignore. The `relative_position_index` entries are non-persistent buffers
the checkpoint still ships. The final `swin.layernorm` is not in the checkpoint at all, and
`transformers` initialises it to the LayerNorm identity affine — but OneFormer consumes the
per-stage feature maps, not the Swin `sequence_output`, so its result is discarded.
Multiplying those weights by 7 leaves the predicted masks bit-identical.

## Determinism

The estimation path is deterministic by default: the same mask and depth map
always produce the same `WidthResult` and the same clearances.

Two steps draw random samples and both take an explicit `seed` (default `0`):

- `processing.geometry.compute_width(..., seed=0)` — jitter applied to the
  near-perpendicular Δu target.
- `processing.refinement.refine_sidewalk_mask(..., seed=0)` — RANSAC sampling for
  the bottom curb line, forwarded to `fill_between_independent_lines` and
  `fit_line_ransac`.

Pass `seed=None` to either one to opt into non-deterministic behaviour, for
example when quantifying the sensitivity of an estimate.

Obstacle overlay colours are derived from a BLAKE2b digest of the label rather
than `hash()`, so they do not change with `PYTHONHASHSEED` between runs.

Model backends are a separate matter: GPU kernel selection and model downloads
are not controlled by these seeds. Record the backend, the model variant, and the
weight cache location alongside any published measurement.

## Metric Scale for Relative Depth Back-Ends

Width estimation reads the depth map as metres. ZoeDepth reports metres directly
(`is_metric = True`) and is passed through untouched.

MiDaS does not. It emits affine-invariant *inverse* depth, so the pipeline has two
things to undo before the map means anything:

1. `models.midas` inverts the disparity into a relative depth map, normalised so the
   nearest surfaces sit near 1.0 and the far field is capped at 100x that distance.
2. `processing.geometry.to_metric_depth` recovers the missing metres-per-unit factor
   by fitting a ground plane (RANSAC) to the lowest sidewalk pixels that carry depth.
   If the depth map were already metric that plane would sit `CAM_HEIGHT_M` (1.75 m)
   from the optical centre, and the ratio gives the factor.

When the fit lacks support the behaviour depends on how the run is configured:

| Configuration | Result |
| --- | --- |
| `--fallback-scale` set (CLI default `0.075`) | the constant is used instead |
| `--force-fallback` | the plane fit is skipped entirely |
| no fallback (Web API default) | the depth path is dropped for that frame and the width comes from geometry alone |

The last row is deliberate: an unscaled depth map would report widths in an arbitrary
unit that looks like metres. The correct value for `--fallback-scale` depends on the
back-end's output convention, so re-derive it whenever the depth model changes.

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
sidewalk-ai --image generic/images/streetview_id1_heading0.jpg `
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
