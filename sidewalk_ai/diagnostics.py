from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

CORE_PACKAGES = [
    "joblib",
    "numpy",
    "opencv-python",
    "pydantic",
    "pydantic-settings",
    "requests",
    "scipy",
]

ML_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "Pillow",
    "matplotlib",
]

BACKEND_MODULES = {
    "detectron2": "detectron2",
    "oneformer_transformers": "transformers",
    "deeplab_local": "network",
    "zoedepth_local": "zoedepth",
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _package_version(name: str) -> Check:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return Check(name=name, ok=False, detail="not installed")
    return Check(name=name, ok=True, detail=version)


def _module_available(name: str, module: str) -> Check:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return Check(name=name, ok=False, detail=f"module '{module}' not importable")
    origin = spec.origin or "namespace/package"
    return Check(name=name, ok=True, detail=origin)


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    if importlib.util.find_spec("torch") is None:
        return info
    try:
        import torch

        info.update(
            {
                "installed": True,
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def collect_diagnostics() -> dict[str, Any]:
    env_keys = [
        "GOOGLE_API_KEY",
        "SWAI_DEPTH",
        "SWAI_DEBUG",
        "SWAI_LOG_LEVEL",
        "SWAI_LOG_FORMAT",
        "SWAI_LOG_FILE",
        "CUDA_HOME",
    ]
    env = {
        key: ("<set>" if key == "GOOGLE_API_KEY" and os.getenv(key) else os.getenv(key))
        for key in env_keys
        if os.getenv(key) is not None
    }
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "environment": env,
        "packages": [asdict(_package_version(pkg)) for pkg in CORE_PACKAGES + ML_PACKAGES],
        "backends": [
            asdict(_module_available(name, module))
            for name, module in sorted(BACKEND_MODULES.items())
        ],
        "torch": _torch_info(),
    }


def _print_text(report: dict[str, Any]) -> None:
    print("Sidewalk AI diagnostics")
    print(f"Python: {report['python']['version'].split()[0]} ({report['python']['executable']})")
    print(
        "Platform: "
        f"{report['platform']['system']} {report['platform']['release']} "
        f"{report['platform']['machine']}"
    )
    torch = report["torch"]
    if torch.get("installed"):
        print(
            "Torch: "
            f"{torch.get('version')} | CUDA available={torch.get('cuda_available')} "
            f"| CUDA={torch.get('cuda_version')}"
        )
        if torch.get("device_name"):
            print(f"GPU: {torch['device_name']}")
    else:
        print("Torch: not installed")

    print("\nPackages:")
    for item in report["packages"]:
        status = "ok" if item["ok"] else "missing"
        print(f"  {item['name']:<18} {status:<8} {item['detail']}")

    print("\nBackends:")
    for item in report["backends"]:
        status = "ok" if item["ok"] else "missing"
        print(f"  {item['name']:<22} {status:<8} {item['detail']}")

    if report["environment"]:
        print("\nEnvironment:")
        for key, value in report["environment"].items():
            print(f"  {key}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Sidewalk AI runtime dependencies.")
    parser.add_argument("--json", action="store_true", help="Write diagnostics as JSON")
    args = parser.parse_args(argv)

    report = collect_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
