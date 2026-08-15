# Architecture

`sidewalk_ai` is the maintained package boundary. Code outside that directory is treated as
research material, sample data, or third-party source.

## Main Flow

```text
CLI/API request
  -> request normalization
  -> StreetViewClient image fetch or local image load
  -> Segmenter adapter
  -> mask refinement
  -> depth estimator adapter
  -> width and clearance geometry
  -> accessibility metrics and optional debug artifacts
```

## Package Boundaries

- `sidewalk_ai.core` owns orchestration. It should not import concrete model libraries directly.
- `sidewalk_ai.models` owns adapters around external ML backends.
- `sidewalk_ai.processing` owns deterministic numerical logic and should be the easiest layer to test.
- `sidewalk_ai.io` owns filesystem, network, image decoding, and geospatial I/O.
- `sidewalk_ai.api` owns request normalization and response-oriented serialization helpers.
- `sidewalk_ai.cli` owns command-line parsing and presentation.
- `sidewalk_ai.webapi` owns the HTTP surface: schemas, the pipeline registry, and the
  inference semaphore. It holds no estimation logic. See `docs/webapi.md`.
- `sidewalk_ai.labels` is a dependency-free leaf shared by `io` and `processing`, so
  neither has to import the other for label parsing.

## Dependency Rules

- Core orchestration depends on protocols or adapters, not on Detectron2, OneFormer, or ZoeDepth internals.
- Tests in `sidewalk_ai/tests` should avoid real network calls, GPU work, and model downloads by default.
- Generated debug outputs should go to `debug_out/` or an explicit output directory.
- New heavyweight experiments should start under `prototype/` and move into `sidewalk_ai/` only after the
  API, tests, and ownership are clear.

## Configuration

Runtime configuration is environment-driven. `.env.example` documents supported keys; `.env` remains local.
Google API credentials are validated only before an API call, which keeps imports and unit tests deterministic.
