import argparse
import os
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Possion.poisson_cloner import PoissonCloner
from utils import _read_color_float64, _read_mask_bool, naive_copy_paste
from utils import _save_color, _save_mask_png, _save_side_by_side, _write_text
from utils import save_zoom_comparison, _choose_zoom_centers



def _mask_bbox(mask: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + 1 + margin, mask.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, mask.shape[1])
    return y0, y1, x0, x1


def _crop_center(mask: np.ndarray, half_size: int) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    else:
        cy, cx = int(np.round(ys.mean())), int(np.round(xs.mean()))
    y0 = max(cy - half_size, 0)
    y1 = min(cy + half_size, mask.shape[0])
    x0 = max(cx - half_size, 0)
    x1 = min(cx + half_size, mask.shape[1])
    return y0, y1, x0, x1



def _run_seamless(src_path: str, target_path: str, mask_path: str) -> Tuple[np.ndarray, dict]:
    cloner = PoissonCloner(src_path, target_path, mask_path)
    poisson, info, solve_time_s = cloner.solve(mode="seamless")
    return poisson, {
        "N": info.n_unknowns,
        "nnzA": info.nnz,
        "build_time_s": info.build_time_s,
        "solve_time_s": solve_time_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="figs/poisson_source.png")
    parser.add_argument("--target_path", type=str, default="figs/poisson_target.png")
    parser.add_argument("--mask_path", type=str, default="figs/poisson_mask.png")
    parser.add_argument("--output_dir", type=str, default="results/results_task4")
    parser.add_argument("--brightness_scale", type=float, default=1.8)
    parser.add_argument("--dilate_iters", type=int, default=25)
    parser.add_argument("--zoom_margin", type=int, default=25)
    parser.add_argument("--zoom_half_size", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    src_path = args.src_path
    target_path = args.target_path
    mask_path = args.mask_path

    src = _read_color_float64(src_path)
    target = _read_color_float64(target_path)
    mask = _read_mask_bool(mask_path)

    _save_color(os.path.join(args.output_dir, "source_original.png"), src)
    _save_color(os.path.join(args.output_dir, "target.png"), target)
    _save_mask_png(os.path.join(args.output_dir, "mask_original.png"), mask)

    poisson_base, base_info = _run_seamless(src_path, target_path, mask_path)
    naive_base = naive_copy_paste(src, target, mask)
    _save_color(os.path.join(args.output_dir, "baseline_naive.png"), naive_base)
    _save_color(os.path.join(args.output_dir, "baseline_poisson.png"), poisson_base)

    src_bright = np.clip(src * float(args.brightness_scale), 0.0, 1.0)
    src_bright_path = os.path.join(args.output_dir, "source_brightness_x{:.2f}.png".format(args.brightness_scale))
    _save_color(src_bright_path, src_bright)

    naive_bright = naive_copy_paste(src_bright, target, mask)
    _save_color(os.path.join(args.output_dir, "case1_naive.png"), naive_bright)

    poisson_bright, case1_info = _run_seamless(src_bright_path, target_path, mask_path)
    _save_color(os.path.join(args.output_dir, "case1_poisson.png"), poisson_bright)

    _save_side_by_side(
        poisson_base,
        poisson_bright,
        "Baseline Poisson (Original Source)",
        "Poisson (Brightness x{:.2f})".format(args.brightness_scale),
        os.path.join(args.output_dir, "case1_global_compare.png"),
    )

    center1, center2 = _choose_zoom_centers(src, target, mask)
    panel_titles_1 = ("Baseline Poisson", "Brightness x{:.2f}".format(args.brightness_scale))
    save_zoom_comparison(
        poisson_base,
        poisson_bright,
        mixed_img=None,
        center=center2,
        half_size=int(args.zoom_half_size),
        out_path=os.path.join(args.output_dir, "case1_zoom_boundary.png"),
        title="Case 1 Zoom Area",
        panel_titles=panel_titles_1,
    )
    save_zoom_comparison(
        poisson_base,
        poisson_bright,
        mixed_img=None,
        center=center1,
        half_size=int(args.zoom_half_size),
        out_path=os.path.join(args.output_dir, "case1_zoom_inside.png"),
        title="Case 1 Zoom Area",
        panel_titles=panel_titles_1,
    )

    from scipy.ndimage import binary_dilation

    dilated = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(args.dilate_iters))
    mask_dilated_path = os.path.join(args.output_dir, "mask_dilated_{}px.png".format(args.dilate_iters))
    _save_mask_png(mask_dilated_path, dilated)

    naive_dilated = naive_copy_paste(src, target, dilated)
    _save_color(os.path.join(args.output_dir, "case2_naive.png"), naive_dilated)

    poisson_dilated, case2_info = _run_seamless(src_path, target_path, mask_dilated_path)
    _save_color(os.path.join(args.output_dir, "case2_poisson.png"), poisson_dilated)

    _save_side_by_side(
        poisson_base,
        poisson_dilated,
        "Baseline Poisson (Original Mask)",
        "Poisson (Dilated Mask {}px)".format(args.dilate_iters),
        os.path.join(args.output_dir, "case2_global_compare.png"),
    )

    center1_2, center2_2 = _choose_zoom_centers(src, target, dilated)
    panel_titles_2 = ("Baseline Poisson", "Dilated Mask {}px".format(args.dilate_iters))
    save_zoom_comparison(
        poisson_base,
        poisson_dilated,
        mixed_img=None,
        center=center2_2,
        half_size=int(args.zoom_half_size),
        out_path=os.path.join(args.output_dir, "case2_zoom_boundary.png"),
        title="Case 2 Zoom Area",
        panel_titles=panel_titles_2,
    )
    save_zoom_comparison(
        poisson_base,
        poisson_dilated,
        mixed_img=None,
        center=center1_2,
        half_size=int(args.zoom_half_size),
        out_path=os.path.join(args.output_dir, "case2_zoom_inside.png"),
        title="Case 2 Zoom Area",
        panel_titles=panel_titles_2,
    )

    summary = []
    summary.append("Baseline: N={N}, nnz(A)={nnzA}, build_time_s={build_time_s:.6f}, solve_time_s={solve_time_s:.6f}".format(**base_info))
    summary.append("Case1 (brightness x{:.2f}): N={N}, nnz(A)={nnzA}, build_time_s={build_time_s:.6f}, solve_time_s={solve_time_s:.6f}".format(float(args.brightness_scale), **case1_info))
    summary.append("Case2 (dilate {}px): N={N}, nnz(A)={nnzA}, build_time_s={build_time_s:.6f}, solve_time_s={solve_time_s:.6f}".format(int(args.dilate_iters), **case2_info))
    _write_text(os.path.join(args.output_dir, "runtime_summary.txt"), "\n".join(summary) + "\n")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
