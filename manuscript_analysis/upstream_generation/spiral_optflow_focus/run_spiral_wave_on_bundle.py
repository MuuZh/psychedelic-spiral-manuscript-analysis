#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiral_wave_detector import SpiralWaveDetector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Detect spiral-wave foci for one frame from bundle/phase_cube.npy "
            "and save an overlay figure."
        )
    )
    p.add_argument("--bundle", type=Path, required=True, help="Bundle directory containing phase_cube.npy")
    p.add_argument("--frame-idx", type=int, required=True, help="Target frame index")
    p.add_argument("--alpha", type=float, default=0.1, help="Horn-Schunck smoothness")
    p.add_argument("--beta", type=float, default=10.0, help="Charbonnier beta")
    p.add_argument("--max-iter", type=int, default=200, help="Max Horn-Schunck iterations")
    p.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance")
    p.add_argument(
        "--winding-radius",
        type=int,
        default=2,
        help="Square-ring radius in pixels for Poincare index validation",
    )
    p.add_argument(
        "--winding-min",
        type=float,
        default=0.8,
        help="Minimum winding number required to accept a spiral focus",
    )
    p.add_argument(
        "--merge-distance",
        type=float,
        default=2.0,
        help="Merge stable optical-flow foci within this grid-pixel distance",
    )
    p.add_argument(
        "--include-unstable",
        action="store_true",
        help="Also show unstable foci; by default only stable foci are detected",
    )
    p.add_argument(
        "--quiver-step",
        type=int,
        default=8,
        help="Grid-pixel stride for drawing the optical-flow vector field",
    )
    p.add_argument(
        "--quiver-scale",
        type=float,
        default=0.35,
        help="Matplotlib quiver scale; smaller values draw longer arrows",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/spiral_optflow_focus"),
        help="Directory for output image/json",
    )
    p.add_argument("--phase-cmap", type=str, default="twilight", help="Matplotlib colormap for phase")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    bundle = args.bundle
    cube_path = bundle / "phase_cube.npy"
    if not cube_path.exists():
        raise FileNotFoundError(f"Missing phase cube: {cube_path}")

    phase_cube = np.load(cube_path)
    frame_idx = int(args.frame_idx)
    if frame_idx < 0 or frame_idx >= phase_cube.shape[2]:
        raise IndexError(f"frame_idx={frame_idx} out of range [0, {phase_cube.shape[2] - 1}]")

    detector = SpiralWaveDetector(
        alpha=args.alpha,
        beta=args.beta,
        max_iter=args.max_iter,
        tol=args.tol,
        winding_radius=args.winding_radius,
        winding_min=args.winding_min,
        stable_only=not args.include_unstable,
        merge_distance=args.merge_distance,
    )
    u2d, v2d = detector.compute_velocity_frame(phase_cube, frame_idx=frame_idx)
    spirals = detector.detect_from_phase_cube(phase_cube, frame_idx=frame_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = bundle.name

    payload = {
        "bundle": str(bundle.resolve()),
        "frame_idx": frame_idx,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "winding_radius": int(args.winding_radius),
        "winding_min": float(args.winding_min),
        "stable_only": bool(not args.include_unstable),
        "merge_distance": float(args.merge_distance),
        "quiver_step": int(args.quiver_step),
        "quiver_scale": float(args.quiver_scale),
        "num_spirals": len(spirals),
        "spirals": [
            {
                "row": float(s.row),
                "col": float(s.col),
                "determinant": float(s.determinant),
                "trace": float(s.trace),
                "stable_focus": bool(s.stable),
                "winding_number": float(s.winding_number),
            }
            for s in spirals
        ],
    }
    json_path = args.output_dir / f"{bundle_name}_frame{frame_idx:04d}_spiral_focus.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    phase = phase_cube[:, :, frame_idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(phase, cmap=args.phase_cmap, origin="lower", vmin=-np.pi, vmax=np.pi)

    yy, xx = np.mgrid[0:phase.shape[0], 0:phase.shape[1]]
    quiver_step = max(int(args.quiver_step), 1)
    sample = (
        np.isfinite(u2d)
        & np.isfinite(v2d)
        & ((yy % quiver_step) == 0)
        & ((xx % quiver_step) == 0)
    )
    if np.any(sample):
        ax.quiver(
            xx[sample],
            yy[sample],
            u2d[sample],
            v2d[sample],
            color="#111111",
            angles="xy",
            scale_units="xy",
            scale=float(args.quiver_scale),
            width=0.002,
            alpha=0.7,
            zorder=2,
        )

    if spirals:
        rows = np.array([s.row for s in spirals], dtype=float)
        cols = np.array([s.col for s in spirals], dtype=float)
        stable_mask = np.array([s.stable for s in spirals], dtype=bool)

        if np.any(stable_mask):
            ax.scatter(
                cols[stable_mask],
                rows[stable_mask],
                s=80,
                c="#00E676",
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                label="Stable focus",
                zorder=3,
            )
        if np.any(~stable_mask):
            ax.scatter(
                cols[~stable_mask],
                rows[~stable_mask],
                s=80,
                c="#FF5252",
                marker="x",
                linewidths=1.4,
                label="Unstable focus",
                zorder=3,
            )
        ax.legend(loc="upper right", frameon=True)

    ax.set_title(f"Spiral foci overlay | {bundle_name} | frame={frame_idx} | n={len(spirals)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Phase (rad)")
    fig.tight_layout()

    png_path = args.output_dir / f"{bundle_name}_frame{frame_idx:04d}_spiral_focus_overlay.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Detected focus count: {len(spirals)}")
    print(f"JSON: {json_path}")
    print(f"Overlay: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
