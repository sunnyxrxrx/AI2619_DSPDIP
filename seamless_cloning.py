import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _read_color_float64(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_f = img.astype(np.float64)
        if img_f.max() > 1.0:
            img_f /= 255.0
        return np.clip(img_f, 0.0, 1.0)
    except Exception:
        img = mpimg.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img_f = img.astype(np.float64)
        if img_f.max() > 1.0:
            img_f /= 255.0
        return np.clip(img_f, 0.0, 1.0)


def _read_mask_bool(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"Failed to read mask: {path}")
        m_f = m.astype(np.float64)
        if m_f.max() > 1.0:
            m_f /= 255.0
        return m_f > 0.5
    except Exception:
        m = mpimg.imread(path)
        if m.ndim == 3:
            m = m[..., 0]
        m_f = m.astype(np.float64)
        if m_f.max() > 1.0:
            m_f /= 255.0
        return m_f > 0.5


def naive_copy_paste(src: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Baseline：直接把 source 的 mask 区域像素覆盖到 target 上（不做任何梯度域融合）
    out = target.copy()
    out[mask] = src[mask]
    return out


def _mask_bbox(mask: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + 1 + margin, mask.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, mask.shape[1])
    return y0, y1, x0, x1


def save_boundary_zoom_comparison(
    naive_img: np.ndarray,
    poisson_img: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    margin: int = 20,
) -> None:
    # 用 mask 的包围盒做局部放大，重点展示边界缝隙差异
    y0, y1, x0, x1 = _mask_bbox(mask, margin=margin)
    naive_crop = naive_img[y0:y1, x0:x1, :]
    poisson_crop = poisson_img[y0:y1, x0:x1, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].imshow(np.clip(naive_crop, 0.0, 1.0))
    axes[0].set_title("Naive Copy-Paste (Zoom)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(poisson_crop, 0.0, 1.0))
    axes[1].set_title("Poisson Seamless Cloning (Zoom)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@dataclass
class BuildInfo:
    n_unknowns: int
    nnz: int
    build_time_s: float


class PoissonCloner:
    def __init__(self, src_path: str, target_path: str, mask_path: str):
        self.src_path = src_path
        self.target_path = target_path
        self.mask_path = mask_path

        self.src = _read_color_float64(src_path)
        self.target = _read_color_float64(target_path)
        self.mask = _read_mask_bool(mask_path)

        if self.src.shape[:2] != self.target.shape[:2]:
            raise ValueError("Source and target must have the same spatial shape.")
        if self.mask.shape != self.target.shape[:2]:
            raise ValueError("Mask must have the same spatial shape as images.")

        h, w = self.mask.shape
        self.h = h
        self.w = w

        # 映射约束（硬性要求）：mask 内每个像素对应一个未知量，A 的规模必须是 N×N
        # im2var[y, x] = i 表示像素 (y, x) 对应线性系统中的第 i 个未知量；mask 外为 -1
        self.im2var = np.full((h, w), -1, dtype=np.int32)
        ys, xs = np.nonzero(self.mask)
        ids = np.arange(ys.size, dtype=np.int32)
        self.im2var[ys, xs] = ids
        self.mask_ys = ys.astype(np.int32)
        self.mask_xs = xs.astype(np.int32)
        self.n = int(ys.size)

        self.a: Optional[sp.csr_matrix] = None
        self.build_info: Optional[BuildInfo] = None

    def build_matrix_A(self) -> BuildInfo:
        if self.a is not None and self.build_info is not None:
            return self.build_info

        t0 = time.perf_counter()
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for y, x in zip(self.mask_ys, self.mask_xs):
            i = int(self.im2var[y, x])
            degree = 0

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yn = y + dy
                xn = x + dx
                if yn < 0 or yn >= self.h or xn < 0 or xn >= self.w:
                    continue
                degree += 1
                j = int(self.im2var[yn, xn])
                if j >= 0:
                    # 4-connected Laplacian：对 mask 内邻居写入 -1
                    rows.append(i)
                    cols.append(j)
                    data.append(-1.0)

            # 对角项系数为该像素在图像中的有效邻居数（通常是 4；靠近图像边缘时可能 < 4）
            rows.append(i)
            cols.append(i)
            data.append(float(degree))

        a = sp.coo_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(self.n, self.n),
        ).tocsr()
        t1 = time.perf_counter()

        self.a = a
        self.build_info = BuildInfo(
            n_unknowns=self.n,
            nnz=int(a.nnz),
            build_time_s=float(t1 - t0),
        )
        return self.build_info

    def _build_b_channel(self, ch: int) -> np.ndarray:
        b = np.zeros(self.n, dtype=np.float64)
        g = self.src[..., ch]
        f_star = self.target[..., ch]

        # 对每个 mask 内像素 p：
        # 4 f_p - sum_{q in N_p ∩ W} f_q = sum_{q in N_p ∩ ∂W} f*_q + sum_{q in N_p} v_{pq}
        # importing gradients 的 guidance：v_{pq} = g_p - g_q（注意方向与正负号）
        for y, x in zip(self.mask_ys, self.mask_xs):
            i = int(self.im2var[y, x])
            gp = float(g[y, x])
            rhs = 0.0

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yn = y + dy
                xn = x + dx
                if yn < 0 or yn >= self.h or xn < 0 or xn >= self.w:
                    continue

                # sum_{q in N_p} v_{pq}
                rhs += gp - float(g[yn, xn])
                if self.im2var[yn, xn] < 0:
                    # q 不在 mask 内时，等价于 Dirichlet：把已知的 f*_q 挪到右端项
                    rhs += float(f_star[yn, xn])

            b[i] = rhs
        return b

    def solve_cloning(self) -> Tuple[np.ndarray, BuildInfo, float]:
        info = self.build_matrix_A()
        if self.a is None:
            raise RuntimeError("Matrix A is not built.")

        t0 = time.perf_counter()
        out = self.target.copy()
        # A 对三个通道相同，只需构建一次；每个通道 RHS 不同，分别求解
        for ch in range(3):
            b = self._build_b_channel(ch)
            x = spsolve(self.a, b).astype(np.float64)
            out[..., ch][self.mask] = x
        out = np.clip(out, 0.0, 1.0)
        t1 = time.perf_counter()
        return out, info, float(t1 - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="poisson_source.png")
    parser.add_argument("--target_path", type=str, default="poisson_target.png")
    parser.add_argument("--mask_path", type=str, default="poisson_mask.png")
    parser.add_argument("--output_dir", type=str, default="results_task2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cloner = PoissonCloner(args.src_path, args.target_path, args.mask_path)
    naive = naive_copy_paste(cloner.src, cloner.target, cloner.mask)
    plt.imsave(os.path.join(args.output_dir, "naive_copy_paste.png"), naive)

    poisson, info, solve_time_s = cloner.solve_cloning()
    plt.imsave(os.path.join(args.output_dir, "poisson_seamless_cloning.png"), poisson)

    save_boundary_zoom_comparison(
        naive,
        poisson,
        cloner.mask,
        os.path.join(args.output_dir, "boundary_zoom_comparison.png"),
    )

    print(
        "N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}, solve_time_s={st:.6f}".format(
            n=info.n_unknowns, nnz=info.nnz, bt=info.build_time_s, st=solve_time_s
        )
    )


if __name__ == "__main__":
    main()
