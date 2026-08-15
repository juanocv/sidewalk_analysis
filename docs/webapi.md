# Web API

The package ships a FastAPI application that exposes the same pipeline the CLI
runs. It is the integration surface for external systems; the CLI remains the
tool for local experiments.

## Running

The API needs both the `api` extra and a working model stack:

```bash
python -m pip install -e ".[api,ml]"
uvicorn sidewalk_ai.webapi:app --host 127.0.0.1 --port 8000
```

Set `GOOGLE_API_KEY` first — every endpoint fetches Street View imagery.

Interactive schema documentation is served at `http://127.0.0.1:8000/docs`, and
the raw OpenAPI document at `/openapi.json`. Those are generated from the code,
so they are always current; this page covers the parts the schema cannot express.

Startup loads the OneFormer segmenter plus the depth back-end named by
`SWAI_DEPTH`, so the first boot is slow and the process holds the weights for its
lifetime. Other depth back-ends and ZoeDepth variants are loaded lazily the first
time a request asks for one.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/ping` | Liveness probe. Returns `{"ok": true}` without touching the models. |
| `POST` | `/analyse/single` | One heading, one measurement. |
| `POST` | `/analyse/multi` | Samples several headings per side and aggregates. |

### `POST /analyse/single`

Provide either `address` or `lat` + `lon`; coordinates win when both are given.

```jsonc
{
  "lat": -23.678479,          // or "address": "Av. Paulista 1578, São Paulo"
  "lon": -46.559621,
  "heading": 0,               // 0-359
  "pitch": -10,               // -90..90, must match how the frame was captured
  "fov": 90,                  // 10..120
  "depth": "zoe",             // "zoe" | "midas"
  "zoe_variant": null,        // "ZoeD_N" | "ZoeD_K" | "ZoeD_NK"
  "refine": true,             // false feeds the raw segmenter mask downstream
  "force_fallback": false,    // non-metric depth only; see reproducibility.md
  "fallback_scale": null,     // metres-per-unit, non-metric depth only
  "return_mask": false,       // include base64 PNG overlays in the response
  "min_clear": 1.20           // NBR 9050 threshold used for the rating
}
```

Response:

```jsonc
{
  "width_m": 2.95,
  "margin_m": 0.31,
  "clearances": [
    {"label": "tree#2:base1", "L_m": 1.35, "R_m": 1.35, "total_m": 2.71, "obs_width": 0.24}
  ],
  "gsv_png_b64": null,               // populated when return_mask = true
  "overlay_sidewalk_png_b64": null,
  "overlay_obstacle_png_b64": null,
  "accessibility": {
    "min_clear_required_m": 1.2,
    "global_stats": {"total_obstacles": 2, "meets_ratio": 0.75, "rating": "III", "...": "..."},
    "per_type": {"tree": {"count": 1, "...": "..."}}
  }
}
```

`rating` is `"I"`, `"II"` or `"III"`, derived from the median free corridor
against `min_clear`; the intermediate band is `SWAI_RANK_MID_RATIO * min_clear`.

### `POST /analyse/multi`

Same body minus `heading` (it is ignored: the pipeline finds the street centre
itself and samples headings around it). The response is a compact summary:

- `multi_metadata.n_headings` — how many headings survived on each side.
- `per_side.LEFT` / `per_side.RIGHT` — `median_width`, `width_range_m`,
  `corridor` and `obstacles` for that side.
- `all_views` — the same corridor block plus `width_range_m` pooled over both sides.
- `per_heading` — one raw entry per analysed heading, including its clearances.
- `samples_left` / `samples_right` — base64 PNG triplets for up to three sampled
  headings, only when `return_mask` is true.

A heading whose refinement fails is skipped rather than failing the request, so
`n_headings` can be lower than the number attempted.

## Status codes

| Code | Meaning |
| --- | --- |
| `400` | Unknown `zoe_variant`. |
| `422` | Neither address nor lat/lon supplied, or a field failed validation. |
| `404` | The imagery for the request could not be found. |
| `500` | Multi-view accessibility aggregation failed; details are in the server log. |
| `503` | Depth back-end not installed, or the inference queue timed out. |

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `GOOGLE_API_KEY` | — | Required; every request fetches Street View. |
| `SWAI_DEPTH` | `zoe` | Depth back-end warmed at startup and used when a request omits `depth`. |
| `SWAI_ZOE_VARIANT` | `zoed_n` | Default ZoeDepth variant. |
| `SWAI_API_MAX_CONCURRENCY` | `1` | Requests allowed inside the models simultaneously. |
| `SWAI_API_QUEUE_TIMEOUT_S` | `300` | How long a request waits for a slot before `503`. |
| `SWAI_API_CORS_ORIGINS` | `*` | Comma-separated allowed origins. |
| `SWAI_RANK_MID_RATIO` | `0.50` | Fraction of `min_clear` separating rating `II` from `I`. |

## Concurrency and cost

Endpoints are synchronous, so Starlette runs them in a worker thread pool and
requests can overlap. The models behind them are not thread-safe and a second
concurrent inference can exhaust GPU memory, so model work is serialised through
a semaphore of size `SWAI_API_MAX_CONCURRENCY`, which defaults to 1. Requests
beyond that queue, and give up with `503` after `SWAI_API_QUEUE_TIMEOUT_S`.

Budget accordingly: `/analyse/multi` performs up to four Street View fetches and
four full inference passes **per side**, all inside one HTTP request. With the
default concurrency of 1 a single multi-view call blocks every other request for
its whole duration. Put it behind a job queue if callers cannot tolerate that.

Each pipeline is cached per `(depth back-end, ZoeDepth variant, refine)`, and the
depth model itself is shared across the `refine` variants, so asking for a new
combination costs one model load and nothing thereafter.

## Security

The API has **no authentication and no rate limiting**. It fetches from a paid
Google API on every call, so an exposed instance is a direct billing risk. Run it
behind a reverse proxy that handles authentication, or bind it to localhost.

`SWAI_API_CORS_ORIGINS` defaults to `*` so the bundled `index.html` works when
opened straight from disk. Narrow it to the front-end's real origin before
deploying anywhere shared.
