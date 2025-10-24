# sidewalk_ai/cli.py
from __future__ import annotations
import json
from pathlib import Path

import typer
from rich import print
from rich.table import Table
from sidewalk_ai.processing.accessibility import (
    compute_single_view_metrics, compute_multiview_metrics
)

import sidewalk_ai as sw

app = typer.Typer(add_completion=False, no_args_is_help=True)
_opts_segmenter  = typer.Option("oneformer", help="back-end: oneformer/detectron2/deeplab")
_opt_seg_task    = typer.Option("panoptic", help="oneformer task: panoptic/instance/semantic")
_opt_dual_pass   = typer.Option(True, help="run panoptic for sidewalk + instance for obstacles")
_opt_fuse_stuff  = typer.Option(False, help="panoptic fuse only STUFF classes")
_opt_device     = typer.Option("cuda", help="'cuda' or 'cpu'")
_opt_min_clear  = typer.Option(1.20, help="Minimum clear walking path (m), ABNT NBR 9050 = 1.20 m")
_opt_metrics_json = typer.Option(None, help="Optional path to write metrics JSON")


def _pipeline(backend: str, device: str,
              seg_task: str, dual_pass: bool, fuse_stuff: bool) -> sw.SidewalkPipeline:
    seg = sw.build_segmenter(
        backend=backend,
        device=device,
        seg_task=seg_task,
        dual_pass_for_instances=dual_pass,
        fuse_stuff=fuse_stuff,
    )
    depth = sw.MidasEstimator(device=device)
    sv    = sw.StreetViewClient()
    return sw.SidewalkPipeline(segmenter=seg, depth=depth, streetview=sv)


# ------------------------------------------------------------------ #
# 1) analyse <address>                                               #
# ------------------------------------------------------------------ #
@app.command(help="Estimate width for a *single* address")
def analyse(
    address: str,
    backend: str = _opts_segmenter,
    device: str = _opt_device,
    min_clear: float = _opt_min_clear,
    metrics_json: Path | None = _opt_metrics_json,
    seg_task: str = _opt_seg_task,
    dual_pass: bool = _opt_dual_pass,
    fuse_stuff: bool = _opt_fuse_stuff,
    save_mask: Path | None = typer.Option(None, help="Optionally save mask as PNG"),
):
    pipe  = _pipeline(backend, device, seg_task, dual_pass, fuse_stuff)
    res   = pipe.analyse_address(address)

    print(f"[bold green]{address}[/] → width = {res.width.width_m:.2f} ± {res.width.margin_m:.2f} m")

    if save_mask:
        import cv2
        cv2.imwrite(str(save_mask), (res.sidewalk_mask * 255).astype("uint8"))

    # print JSON to stdout (machines can parse it)
    data = {
        "width_m": res.width.width_m,
        "margin_m": res.width.margin_m,
        "clearances": [c.__dict__ for c in res.clearances],
    }

    # ----- accessibility metrics (single-view) -----
    acc = compute_single_view_metrics(res.clearances, min_clear_required_m=min_clear)
    data["accessibility"] = {
        "min_clear_required_m": acc.min_clear_required_m,
        "global": acc.global_stats.__dict__,
        "per_type": {k: v.__dict__ for k, v in acc.per_type.items()},
    }

    # pretty print summary
    print(f"[bold cyan]Accessibility (threshold {min_clear:.2f} m)[/]")
    g = acc.global_stats
    print(f"Total obstacles: {g.total_obstacles} | Corridors ≥{min_clear:.2f}m: {g.meets_120m_ratio:.0%} | Rating: [bold]{g.rating}[/]")
    # compact table per type
    if acc.per_type:
        tbl = Table(title="Per-type free path (median, meters) – corridor = pooled L∪R")
        tbl.add_column("Type"); tbl.add_column("Left"); tbl.add_column("Right"); tbl.add_column("Corridor")
        for t, m in acc.per_type.items():
            tbl.add_row(
                t,
                f"{m.free_left_m.get('median', float('nan')):.2f}",
                f"{m.free_right_m.get('median', float('nan')):.2f}",
                f"{m.free_total_m.get('median', float('nan')):.2f}",
            )
        print(tbl)

    # write JSON if requested
    if metrics_json:
        metrics_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"[bold green]Saved metrics → {metrics_json}")

    print(json.dumps(data, ensure_ascii=False, indent=2))

# ------------------------------------------------------------------ #
# 2) multi-view                                                #
# ------------------------------------------------------------------ #
@app.command(help="Analyse one address with multi-view (LEFT/RIGHT) and aggregate accessibility metrics")
def multiview(
    address: str,
    backend: str = _opts_segmenter,
    device: str = _opt_device,
    min_clear: float = _opt_min_clear,
    metrics_json: Path | None = _opt_metrics_json,
    seg_task: str = _opt_seg_task,
    dual_pass: bool = _opt_dual_pass,
    fuse_stuff: bool = _opt_fuse_stuff,
):
    # construir pipeline com os mesmos knobs do single-view
    pipe = _pipeline(backend, device, seg_task, dual_pass, fuse_stuff)
    # usar o novo método específico de multi-view
    left_results, right_results = pipe.analyse_address_multiview(address)

    acc = compute_multiview_metrics(left_results, right_results, min_clear_required_m=min_clear)
    # impressão resumida
    for side in ("LEFT", "RIGHT", "ALL"):
        g = acc[side].global_stats
        avg = g.avg_obstacles_per_view_rounded or g.avg_obstacles_per_view or g.total_obstacles
        print(f"[bold cyan]{side}[/] → Obstacles≈{avg} (avg/view) | "
              f"median(corridor)={g.free_total_m.get('median', float('nan')):.2f} m | "
              f"corridors ≥{min_clear:.2f}m={g.meets_120m_ratio:.0%} | Rating=[bold]{g.rating}[/]")

    # JSON
    out = {
        "LEFT": {
            "min_clear_required_m": acc["LEFT"].min_clear_required_m,
            "global": acc["LEFT"].global_stats.__dict__,
            "per_type": {k: v.__dict__ for k, v in acc["LEFT"].per_type.items()},
        },
        "RIGHT": {
            "min_clear_required_m": acc["RIGHT"].min_clear_required_m,
            "global": acc["RIGHT"].global_stats.__dict__,
            "per_type": {k: v.__dict__ for k, v in acc["RIGHT"].per_type.items()},
        },
        "ALL": {
            "min_clear_required_m": acc["ALL"].min_clear_required_m,
            "global": acc["ALL"].global_stats.__dict__,
            "per_type": {k: v.__dict__ for k, v in acc["ALL"].per_type.items()},
        },
    }
    if metrics_json:
        metrics_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[bold green]Saved metrics → {metrics_json}")

    print(json.dumps(out, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ #
# 3) batch <file>.txt                                                #
# ------------------------------------------------------------------ #
@app.command(help="Analyse *many* addresses from a newline-separated file")
def batch(
    file: Path,
    backend: str = _opts_segmenter,
    device: str = _opt_device,
    seg_task: str = _opt_seg_task,
    dual_pass: bool = _opt_dual_pass,
    fuse_stuff: bool = _opt_fuse_stuff,
    out: Path = typer.Option(Path("results.json"), help="Where to store JSON"),
):
    pipe   = _pipeline(backend, device, seg_task, dual_pass, fuse_stuff)
    addrs  = [l.strip() for l in file.read_text("utf-8").splitlines() if l.strip()]

    results = {}
    for a in addrs:
        res = pipe.analyse_address(a)
        results[a] = {
            "width_m": res.width.width_m,
            "margin_m": res.width.margin_m,
        }
        print(f"✓ {a}")

    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[bold green]Saved → {out}")


if __name__ == "__main__":
    app()