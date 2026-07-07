import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import _read_grayscale_float64, _write_text


def _forward_gradients(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = s.shape
    gx = s[:, 1:w] - s[:, 0 : w - 1]
    gy = s[1:h, :] - s[0 : h - 1, :]
    return gx, gy


def _divergence_backward(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    h = gx.shape[0]
    w = gy.shape[1]
    div = np.zeros((h, w), dtype=np.float64)
    if h < 3 or w < 3:
        return div

    term_x = gx[1 : h - 1, 0 : w - 2] - gx[1 : h - 1, 1 : w - 1]
    term_y = gy[0 : h - 2, 1 : w - 1] - gy[1 : h - 1, 1 : w - 1]
    div[1 : h - 1, 1 : w - 1] = term_x + term_y
    return div


def toy_reconstruct(img_path: str, output_dir: str) -> dict:
    s = _read_grayscale_float64(img_path)
    h, w = s.shape
    n = h * w

    t_build0 = time.perf_counter()
    gx, gy = _forward_gradients(s)
    div_g = _divergence_backward(gx, gy)

    interior_count = max(h - 2, 0) * max(w - 2, 0)
    boundary_count = n - interior_count
    nnz = boundary_count + 5 * interior_count

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    b = np.empty(n, dtype=np.float64)

    k = 0
    for y in range(h):
        base = y * w
        for x in range(w):
            i = base + x
            is_boundary = (y == 0) or (y == h - 1) or (x == 0) or (x == w - 1)
            if is_boundary:
                rows[k] = i
                cols[k] = i
                data[k] = 1.0
                k += 1
                b[i] = s[y, x]
                continue

            rows[k : k + 5] = i
            cols[k : k + 5] = np.array(
                [i, i - w, i + w, i - 1, i + 1], dtype=np.int32
            )
            data[k : k + 5] = np.array([4.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
            k += 5
            b[i] = div_g[y, x]

    if k != nnz:
        raise RuntimeError(f"nnz mismatch: filled={k}, expected={nnz}")

    a = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    build_time_s = float(time.perf_counter() - t_build0)

    t_solve0 = time.perf_counter()
    v = spsolve(a, b).astype(np.float64)
    solve_time_s = float(time.perf_counter() - t_solve0)
    v_img = v.reshape(h, w)

    diff = s - v_img
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10(1.0 / mse))
    max_abs_error = float(np.max(np.abs(diff)))

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    ax.imshow(s, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Original (Grayscale, 0–1)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "original.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
    ax.imshow(np.clip(v_img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Reconstructed (Grayscale, 0–1)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reconstructed.png"), bbox_inches="tight")
    plt.close(fig)

    vmax_err = max(max_abs_error, 1e-12)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=150)
    im = ax.imshow(np.abs(diff), cmap="viridis", vmin=0.0, vmax=vmax_err)
    ax.set_title("Absolute Error (Normalized Intensity)")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error (0–1)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "abs_error.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axes[0].imshow(s, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original (Grayscale, 0–1)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(v_img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Reconstructed (Grayscale, 0–1)")
    axes[1].axis("off")

    im = axes[2].imshow(np.abs(diff), cmap="viridis", vmin=0.0, vmax=vmax_err)
    axes[2].set_title("Absolute Error (Normalized Intensity)")
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error (0–1)")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), bbox_inches="tight")
    plt.close(fig)

    summary = "Toy Reconstruction: N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}, solve_time_s={st:.6f}".format(
        n=n, nnz=int(nnz), bt=build_time_s, st=solve_time_s
    )
    _write_text(os.path.join(output_dir, "runtime_summary.txt"), summary + "\n", append=True)

    return {
        "mse": mse,
        "psnr": psnr,
        "max_abs_error": max_abs_error,
        "height": h,
        "width": w,
        "n_unknowns": n,
        "nnz": int(nnz),
        "build_time_s": build_time_s,
        "solve_time_s": solve_time_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="figs/poisson_toy_reconstruction.png")
    parser.add_argument("--output_dir", type=str, default="results/results_toy_reconstruction")
    args = parser.parse_args()

    metrics = toy_reconstruct(args.img_path, args.output_dir)
    print(
        "MSE={mse:.16e}, PSNR={psnr}, MaxAbsError={mae:.16e}, H={h}, W={w}".format(
            mse=metrics["mse"],
            psnr=metrics["psnr"],
            mae=metrics["max_abs_error"],
            h=metrics["height"],
            w=metrics["width"],
        )
    )
    print(
        "N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}, solve_time_s={st:.6f}".format(
            n=metrics["n_unknowns"],
            nnz=metrics["nnz"],
            bt=metrics["build_time_s"],
            st=metrics["solve_time_s"],
        )
    )


if __name__ == "__main__":
    main()
