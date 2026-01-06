#!/usr/bin/env python3
"""encode_benchmark_vjepa2.py

Encode (only encode) the RGB frames in a Kubric benchmark folder using V-JEPA 2
ViT-L/16 ("vjepa2_vit_large") from https://github.com/facebookresearch/vjepa2.

This script:
- Loads the pretrained encoder + eval preprocessor via `torch.hub`.
- Iterates over `clip_*/rgb/frame_*.png`.
- Encodes each clip into patch features and saves one .npy file per clip.
- Uses all visible GPUs via `torch.nn.DataParallel` (requires batch_size > 1).
- Tracks wall-clock time and reports throughput.

Example:
  python experiments/scripts/encode_benchmark_vjepa2.py \
    --data_dir experiments/output/benchmark_test \
    --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def _ensure_light_dependencies() -> None:
    """Auto-install small, pure-python deps.

    We intentionally do NOT auto-install heavyweight deps like torch/torchvision.
    """

    required = [
        ("numpy", "numpy"),
        ("PIL", "pillow"),
    ]

    def _ensure_pip_available() -> None:
        try:
            import pip  # noqa: F401

            return
        except Exception:
            pass

        # Some system Pythons are shipped without pip.
        try:
            import ensurepip

            ensurepip.bootstrap(upgrade=True)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "pip is not available for this Python interpreter. "
                "Install it (e.g. `sudo apt-get install python3-pip`) or use a venv/conda env. "
                "Then install deps: `pip install numpy pillow`."
            ) from e

    for module_name, pip_name in required:
        try:
            __import__(module_name)
        except ImportError:
            _ensure_pip_available()
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


_ensure_light_dependencies()

import numpy as np


def _require_imports() -> Tuple["torch", "Image"]:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'torch'. Install PyTorch with CUDA for GPU encoding. "
            "See https://pytorch.org/get-started/locally/"
        ) from e

    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'Pillow'. Install with: pip install pillow") from e

    # vjepa2 preprocessor requires torchvision/timm/einops; fail early with a clear hint.
    try:
        import torchvision  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'torchvision' (required by vjepa2 transforms). Install with: pip install torchvision"
        ) from e

    try:
        import timm  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'timm'. Install with: pip install timm") from e

    try:
        import einops  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'einops'. Install with: pip install einops") from e

    return torch, Image


@dataclass(frozen=True)
class ClipItem:
    clip_dir: Path
    clip_id: str
    rgb_dir: Path
    frame_paths: List[Path]


def _discover_clips(data_dir: Path) -> List[ClipItem]:
    clip_items: List[ClipItem] = []
    for clip_dir in sorted(data_dir.glob("clip_*")):
        if not clip_dir.is_dir():
            continue
        rgb_dir = clip_dir / "rgb"
        if not rgb_dir.exists():
            continue
        frame_paths = sorted(rgb_dir.glob("frame_*.png"))
        if not frame_paths:
            continue
        clip_items.append(
            ClipItem(
                clip_dir=clip_dir,
                clip_id=clip_dir.name,
                rgb_dir=rgb_dir,
                frame_paths=frame_paths,
            )
        )
    return clip_items


def _load_clip_as_numpy(frame_paths: List[Path], Image) -> np.ndarray:
    # Returns: T x H x W x C (uint8)
    frames: List[np.ndarray] = []
    for p in frame_paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            frames.append(np.array(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _save_embedding(out_path: Path, tensor, dtype: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().numpy()
    if dtype == "fp16":
        arr = arr.astype(np.float16, copy=False)
    elif dtype == "fp32":
        arr = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    np.save(out_path, arr)


def _batched(items: List[ClipItem], batch_size: int) -> List[List[ClipItem]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode experiments/output/benchmark_test clips using V-JEPA2 ViT-L/16 (encoder only)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("experiments/output/benchmark_test"),
        help="Path to benchmark_test folder containing clip_*/rgb/frame_*.png",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write embeddings (.npy). Default: <data_dir>/embeddings_vjepa2_vitl16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of clips per forward pass. Use >1 to leverage multiple GPUs via DataParallel.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Numpy dtype used when saving embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="V-JEPA2 eval crop_size (ViT-L/16 is 256px by default).",
    )
    parser.add_argument(
        "--log_json",
        type=Path,
        default=None,
        help="Optional path to write a JSON timing report.",
    )
    args = parser.parse_args()

    torch, Image = _require_imports()

    data_dir: Path = args.data_dir
    if args.out_dir is None:
        out_dir = data_dir / "embeddings_vjepa2_vitl16"
    else:
        out_dir = args.out_dir

    clip_items = _discover_clips(data_dir)
    if not clip_items:
        raise SystemExit(f"No clips found under: {data_dir}")

    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if device_count > 0 else "cpu")
    if device.type != "cuda":
        raise SystemExit(
            "No CUDA devices detected. This script is intended to run with GPUs. "
            "If you want CPU encoding anyway, remove this guard."
        )

    # Load preprocessor + model via torch.hub
    # - vjepa2_preprocessor returns eval transforms
    # - vjepa2_vit_large is ViT-L/16 encoder (pretrained)
    preprocessor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor", crop_size=args.crop_size)
    encoder, _predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    model = encoder
    model.eval().to(device)

    if device_count > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel across {device_count} GPUs")
    else:
        print("Using single GPU")

    # Simple manual batching to keep per-item metadata for saving.
    batches = _batched(clip_items, max(1, args.batch_size))

    # Warm-up: one small forward pass (helps stabilize timing on GPU)
    warm_item = batches[0][0]
    warm_video_np = _load_clip_as_numpy(warm_item.frame_paths, Image)
    warm_tensor = preprocessor(warm_video_np)[0]  # C x T x H x W
    warm_x = warm_tensor.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        _ = model(warm_x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_clips = 0
    saved_clips = 0
    skipped_clips = 0

    # Optional granular timing
    io_time = 0.0
    preprocess_time = 0.0
    encode_time = 0.0
    save_time = 0.0

    for batch in batches:
        total_clips += len(batch)

        # Skip logic (all-or-nothing per item)
        batch_to_run: List[ClipItem] = []
        for item in batch:
            out_path = out_dir / f"{item.clip_id}.npy"
            if out_path.exists() and not args.overwrite:
                skipped_clips += 1
            else:
                batch_to_run.append(item)
        if not batch_to_run:
            continue

        # Load + preprocess
        t_io0 = time.perf_counter()
        videos_np = [_load_clip_as_numpy(item.frame_paths, Image) for item in batch_to_run]
        io_time += time.perf_counter() - t_io0

        t_p0 = time.perf_counter()
        tensors = [preprocessor(v_np)[0] for v_np in videos_np]  # each: C x T x H x W
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)  # B x C x T x H x W
        preprocess_time += time.perf_counter() - t_p0

        # Encode
        t_e0 = time.perf_counter()
        with torch.inference_mode():
            out = model(x)
        torch.cuda.synchronize()
        encode_time += time.perf_counter() - t_e0

        # Save
        t_s0 = time.perf_counter()
        for item, emb in zip(batch_to_run, out):
            out_path = out_dir / f"{item.clip_id}.npy"
            _save_embedding(out_path, emb, dtype=args.dtype)
            saved_clips += 1
        save_time += time.perf_counter() - t_s0

        elapsed = time.perf_counter() - t0
        clips_per_s = saved_clips / elapsed if elapsed > 0 else 0.0
        print(
            f"Encoded {saved_clips}/{len(clip_items)} clips "
            f"(skipped {skipped_clips}), {clips_per_s:.2f} clips/s"
        )

    total_time = time.perf_counter() - t0
    overall_cps = saved_clips / total_time if total_time > 0 else 0.0
    print("=" * 60)
    print("V-JEPA2 ENCODING SUMMARY")
    print("=" * 60)
    print(f"Data dir:           {data_dir}")
    print(f"Output dir:         {out_dir}")
    print(f"Clips discovered:   {len(clip_items)}")
    print(f"Clips saved:        {saved_clips}")
    print(f"Clips skipped:      {skipped_clips}")
    print(f"Total wall time:    {total_time:.2f} s")
    print(f"Throughput:         {overall_cps:.2f} clips/s")
    print("--- breakdown (approx wall time) ---")
    print(f"I/O (png decode):   {io_time:.2f} s")
    print(f"Preprocess:         {preprocess_time:.2f} s")
    print(f"Encode (GPU):       {encode_time:.2f} s")
    print(f"Save (.npy):        {save_time:.2f} s")

    if args.log_json is not None:
        report = {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "model": "facebookresearch/vjepa2:vjepa2_vit_large",
            "preprocessor": "facebookresearch/vjepa2:vjepa2_preprocessor",
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device_count": int(device_count),
            "clips_discovered": int(len(clip_items)),
            "clips_saved": int(saved_clips),
            "clips_skipped": int(skipped_clips),
            "total_wall_s": float(total_time),
            "throughput_clips_per_s": float(overall_cps),
            "breakdown_s": {
                "io": float(io_time),
                "preprocess": float(preprocess_time),
                "encode": float(encode_time),
                "save": float(save_time),
            },
            "timestamp_unix": time.time(),
        }
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(json.dumps(report, indent=2))
        print(f"Wrote timing report: {args.log_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
