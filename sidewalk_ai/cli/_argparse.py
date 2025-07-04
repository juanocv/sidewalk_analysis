import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SidewalkAI Playground")
    parser.add_argument("address", nargs="?", help="Free-form address string")
    parser.add_argument("--image", type=Path, help="Analyse an existing JPG/PNG")
    parser.add_argument("--seg",
                        default="oneformer",
                        help="Single back-end or TWO separated by a plus: "
                            "'oneformer', 'detectron2', 'deeplab', "
                            "or e.g. 'oneformer+detectron2'")
    parser.add_argument("--ensemble-method", default="or",
                        choices=["or", "and", "majority"],
                        help="Fusion rule when two back-ends are given")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ckpt", help="Path to DeepLab checkpoint (.pth)")
    parser.add_argument("--deeplab-model", default="deeplabv3plus_resnet101",
                        help="Model ctor name inside your network.modeling "
                            "(e.g. deeplabv3plus_mobilenetv3_large)")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Ignore ground-plane fit; always use fallback scale")
    parser.add_argument("--fallback-scale", type=float, default=0.075,
                        help="Constant metres-per-unit when ground-plane fit "
                            "fails (default 0.075 for 600×400 Street View)")
    parser.add_argument("--debug",  action="store_true",
                        help="Verbose console + composite debug image")
    parser.add_argument("--outdir", type=Path, default=Path("debug_out"),
                        help="Folder where debug PNGs are written")
    return parser