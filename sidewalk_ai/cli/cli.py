# sidewalk_ai/cli.py
from __future__ import annotations
import json
from pathlib import Path

import typer
from rich import print

import sidewalk_ai as sw

app = typer.Typer(add_completion=False, no_args_is_help=True)
_opts_segmenter  = typer.Option("oneformer", help="back-end: oneformer/detectron2/deeplab")
_opt_seg_task    = typer.Option("panoptic", help="oneformer task: panoptic/instance/semantic")
_opt_dual_pass   = typer.Option(True, help="run panoptic for sidewalk + instance for obstacles")
_opt_fuse_stuff  = typer.Option(False, help="panoptic fuse only STUFF classes")
_opt_device     = typer.Option("cuda", help="'cuda' or 'cpu'")


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
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ #
# 2) batch <file>.txt                                                #
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