import argparse
import math
import os
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import _read_grayscale_float64


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
    v = spsolve(a, b).astype(np.float64)
    v_img = v.reshape(h, w)

    diff = s - v_img
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10(1.0 / mse))
    max_abs_error = float(np.max(np.abs(diff)))

    os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, "original.png"), s, cmap="gray", vmin=0.0, vmax=1.0)
    plt.imsave(
        os.path.join(output_dir, "reconstructed.png"),
        np.clip(v_img, 0.0, 1.0),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    vmax_err = max(max_abs_error, 1e-12)
    plt.imsave(
        os.path.join(output_dir, "abs_error.png"),
        np.abs(diff),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_err,
    )

    return {
        "mse": mse,
        "psnr": psnr,
        "max_abs_error": max_abs_error,
        "height": h,
        "width": w,
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


if __name__ == "__main__":
    main()
