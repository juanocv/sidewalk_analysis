import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SidewalkAI Playground")
    parser.add_argument("address", nargs="?", help="Free-form address string")
    parser.add_argument("--image", type=Path, help="Analyse an existing JPG/PNG")
    parser.add_argument(
        "--seg",
        default="oneformer",
        help="Single back-end or TWO separated by a plus: "
        "'oneformer', 'detectron2', 'deeplab', "
        "or e.g. 'oneformer+detectron2'",
    )
    parser.add_argument(
        "--ensemble-method",
        default="or",
        choices=["or", "and", "majority"],
        help="Fusion rule when two back-ends are given",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ckpt", help="Path to DeepLab checkpoint (.pth)")
    parser.add_argument(
        "--deeplab-model",
        default="deeplabv3plus_resnet101",
        help="Model ctor name inside your network.modeling "
        "(e.g. deeplabv3plus_mobilenetv3_large)",
    )
    parser.add_argument("--depth", default="zoe", choices=["midas", "zoe"])
    parser.add_argument(
        "--zoe-variant",
        default="zoed_n",
        choices=["zoed_n", "zoed_k", "zoed_nk"],
        help="ZoeDepth model size",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Ignore ground-plane fit; always use fallback scale",
    )
    parser.add_argument(
        "--fallback-scale",
        type=float,
        default=0.075,
        help="Constant metres-per-unit when ground-plane fit "
        "fails (default 0.075 for 600×400 Street View)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Verbose console + composite debug image"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Defaults to SWAI_LOG_LEVEL or INFO.",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["text", "json"],
        help="Logging format. Defaults to SWAI_LOG_FORMAT or text.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file path for logs. Can also be set with SWAI_LOG_FILE.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("debug_out"), help="Folder where debug PNGs are written"
    )
    parser.add_argument(
        "--lat", type=float, default=None, help="Latitude for coordinate-based analysis"
    )
    parser.add_argument(
        "--lon", type=float, default=None, help="Longitude for coordinate-based analysis"
    )

    # Multi-view vs single-view control (CLI default: multi-view)
    parser.add_argument(
        "--multi-view",
        dest="multi_view",
        action="store_true",
        help="Run multi-angle analysis (default)",
    )
    parser.add_argument(
        "--single-view",
        dest="multi_view",
        action="store_false",
        help="Run single-frame analysis using heading/pitch/fov",
    )
    parser.set_defaults(multi_view=True)

    # Single-view parameters
    parser.add_argument(
        "--heading", type=int, default=0, help="Heading for single-view mode (0-359)"
    )
    parser.add_argument(
        "--pitch", type=int, default=-10, help="Pitch for single-view mode (-90 to 90)"
    )
    parser.add_argument(
        "--fov", type=int, default=90, help="Field-of-view for single-view mode (10-120)"
    )
    parser.add_argument(
        "--return-mask",
        action="store_true",
        help="Return mask and overlay images for single-view runs",
    )

    # Accessibility metrics knobs
    parser.add_argument(
        "--min-clear",
        type=float,
        default=1.20,
        help="Minimum free walking path (meters). "
        "ABNT NBR 9050 recomenda ≥ 1.20 m (default 1.20)",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="If set, write accessibility metrics JSON to this path",
    )
    return parser
