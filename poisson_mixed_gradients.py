import argparse
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

Mode = Literal["seamless", "mixed"]


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
    # Baseline：直接粘贴（不做 Poisson 融合），用于和 Poisson 结果做对比
    out = target.copy()
    out[mask] = src[mask]
    return out


def _crop_box_centered(
    h: int, w: int, cy: int, cx: int, half_size: int
) -> Tuple[int, int, int, int]:
    y0 = max(cy - half_size, 0)
    y1 = min(cy + half_size, h)
    x0 = max(cx - half_size, 0)
    x1 = min(cx + half_size, w)
    return y0, y1, x0, x1


def _choose_zoom_centers(
    src: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (h // 2, w // 2), (h // 2, w // 2)

    g_src = src.mean(axis=2)
    g_tgt = target.mean(axis=2)

    # 区域 1（背景纹理穿透）：在 mask 内找一个“目标梯度显著强于源梯度”的位置
    # 直觉：mixed gradients 会更倾向选用目标的大梯度，使背景纹理穿透到融合区域内部
    score_inside = np.full((h, w), -np.inf, dtype=np.float64)
    dy_dx = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y, x in zip(ys, xs):
        s = 0.0
        for dy, dx in dy_dx:
            yn = int(y + dy)
            xn = int(x + dx)
            if yn < 0 or yn >= h or xn < 0 or xn >= w:
                continue
            gt = float(g_tgt[y, x] - g_tgt[yn, xn])
            gs = float(g_src[y, x] - g_src[yn, xn])
            s += max(abs(gt) - abs(gs), 0.0)
        score_inside[y, x] = s

    y1, x1 = np.unravel_index(int(np.argmax(score_inside)), score_inside.shape)

    # 区域 2（边界过渡自然度）：在 mask 边界找一个“源梯度较大”的位置（通常对应源物体强边缘）
    boundary = mask.copy()
    eroded = mask.copy()
    for dy, dx in dy_dx:
        shifted = np.zeros_like(mask)
        y_src0 = max(0, -dy)
        y_src1 = h - max(0, dy)
        x_src0 = max(0, -dx)
        x_src1 = w - max(0, dx)
        shifted[y_src0 + dy : y_src1 + dy, x_src0 + dx : x_src1 + dx] = mask[
            y_src0:y_src1, x_src0:x_src1
        ]
        eroded &= shifted
    boundary &= ~eroded

    bys, bxs = np.nonzero(boundary)
    if bys.size == 0:
        y2, x2 = int(ys.mean()), int(xs.mean())
    else:
        score_b = np.full((h, w), -np.inf, dtype=np.float64)
        for y, x in zip(bys, bxs):
            s = 0.0
            for dy, dx in dy_dx:
                yn = int(y + dy)
                xn = int(x + dx)
                if yn < 0 or yn >= h or xn < 0 or xn >= w:
                    continue
                gs = float(g_src[y, x] - g_src[yn, xn])
                s += abs(gs)
            score_b[y, x] = s
        y2, x2 = np.unravel_index(int(np.argmax(score_b)), score_b.shape)

    return (int(y1), int(x1)), (int(y2), int(x2))


def save_global_comparison(
    naive_img: np.ndarray,
    seamless_img: np.ndarray,
    mixed_img: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axes[0].imshow(np.clip(naive_img, 0.0, 1.0))
    axes[0].set_title("Naive Copy-Paste")
    axes[0].axis("off")

    axes[1].imshow(np.clip(seamless_img, 0.0, 1.0))
    axes[1].set_title("Source-gradient Poisson")
    axes[1].axis("off")

    axes[2].imshow(np.clip(mixed_img, 0.0, 1.0))
    axes[2].set_title("Mixed-gradient Poisson")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_zoom_comparison(
    naive_img: np.ndarray,
    seamless_img: np.ndarray,
    mixed_img: np.ndarray,
    center: Tuple[int, int],
    half_size: int,
    out_path: str,
    title: str,
) -> None:
    # 把同一个局部区域在三种方法下的结果放到同一张图里，方便报告对比
    h, w = naive_img.shape[:2]
    cy, cx = center
    y0, y1, x0, x1 = _crop_box_centered(h, w, cy, cx, half_size)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
    for ax, img, t in zip(
        axes,
        (naive_img, seamless_img, mixed_img),
        ("Naive Copy-Paste", "Source-gradient Poisson", "Mixed-gradient Poisson"),
    ):
        ax.imshow(np.clip(img[y0:y1, x0:x1, :], 0.0, 1.0))
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(title)
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
        self.src = _read_color_float64(src_path)
        self.target = _read_color_float64(target_path)
        self.mask = _read_mask_bool(mask_path)

        if self.src.shape[:2] != self.target.shape[:2]:
            raise ValueError("Source and target must have the same spatial shape.")
        if self.mask.shape != self.target.shape[:2]:
            raise ValueError("Mask must have the same spatial shape as images.")

        self.h, self.w = self.mask.shape

        # 映射系统（硬性要求）：mask 内每个像素一个未知量，因此 A 的规模为 N×N
        # im2var[y, x] ∈ [0, N-1]；mask 外为 -1（表示该像素是已知量，来自 target 的 Dirichlet 边界）
        self.im2var = np.full((self.h, self.w), -1, dtype=np.int32)
        ys, xs = np.nonzero(self.mask)
        self.im2var[ys, xs] = np.arange(ys.size, dtype=np.int32)
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

        # 离散拉普拉斯：对 mask 内每个像素 p，左端是 degree(p)*f_p - sum_{q in N_p ∩ W} f_q
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
                    rows.append(i)
                    cols.append(j)
                    data.append(-1.0)

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

    def _build_b_channel(self, ch: int, mode: Mode) -> np.ndarray:
        b = np.zeros(self.n, dtype=np.float64)

        g = self.src[..., ch]
        f_star = self.target[..., ch]

        # 方程（对每个 p ∈ W）：
        # degree(p) * f_p - sum_{q in N_p ∩ W} f_q = sum_{q in N_p ∩ ∂W} f*_q + sum_{q in N_p} v_{pq}
        # v_{pq} 的定义由 mode 决定，注意方向始终为 “p - q”，避免符号混乱
        for y, x in zip(self.mask_ys, self.mask_xs):
            i = int(self.im2var[y, x])
            rhs = 0.0

            gp = float(g[y, x])
            tp = float(f_star[y, x])

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yn = y + dy
                xn = x + dx
                if yn < 0 or yn >= self.h or xn < 0 or xn >= self.w:
                    continue

                gq = float(g[yn, xn])
                tq = float(f_star[yn, xn])

                if mode == "seamless":
                    # 实验 B：importing gradients，v_{pq} = g_p - g_q
                    v_pq = gp - gq
                else:
                    # 实验 C：mixed gradients
                    # 在每条有向边 (p -> q) 上对比 |target 梯度| 与 |source 梯度|，选更大的那个
                    grad_t = tp - tq
                    grad_s = gp - gq
                    if abs(grad_t) > abs(grad_s):
                        v_pq = grad_t
                    else:
                        v_pq = grad_s

                rhs += v_pq

                # 若 q 不在 mask 内，则 f_q 已知且固定为 f*_q，作为 Dirichlet 边界项移到右端
                if self.im2var[yn, xn] < 0:
                    rhs += tq

            b[i] = rhs
        return b

    def solve(self, mode: Mode) -> Tuple[np.ndarray, BuildInfo, float]:
        info = self.build_matrix_A()
        if self.a is None:
            raise RuntimeError("Matrix A is not built.")

        t0 = time.perf_counter()
        out = self.target.copy()

        # 通道规范：A 共用，b 按通道分别构建并独立求解
        for ch in range(3):
            b = self._build_b_channel(ch, mode=mode)
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
    parser.add_argument("--output_dir", type=str, default="results_task3")
    parser.add_argument("--zoom_half_size", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cloner = PoissonCloner(args.src_path, args.target_path, args.mask_path)

    naive = naive_copy_paste(cloner.src, cloner.target, cloner.mask)
    plt.imsave(os.path.join(args.output_dir, "naive_copy_paste.png"), naive)

    seamless, info, t_seamless = cloner.solve(mode="seamless")
    plt.imsave(os.path.join(args.output_dir, "source_gradient_poisson.png"), seamless)

    mixed, _, t_mixed = cloner.solve(mode="mixed")
    plt.imsave(os.path.join(args.output_dir, "mixed_gradient_poisson.png"), mixed)

    save_global_comparison(
        naive,
        seamless,
        mixed,
        os.path.join(args.output_dir, "global_comparison.png"),
    )

    center1, center2 = _choose_zoom_centers(cloner.src, cloner.target, cloner.mask)
    save_zoom_comparison(
        naive,
        seamless,
        mixed,
        center=center1,
        half_size=args.zoom_half_size,
        out_path=os.path.join(args.output_dir, "zoom_area_1.png"),
        title="Zoom Area 1: Background Texture Penetration",
    )
    save_zoom_comparison(
        naive,
        seamless,
        mixed,
        center=center2,
        half_size=args.zoom_half_size,
        out_path=os.path.join(args.output_dir, "zoom_area_2.png"),
        title="Zoom Area 2: Boundary Transition",
    )

    print(
        "N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}".format(
            n=info.n_unknowns, nnz=info.nnz, bt=info.build_time_s
        )
    )
    print("SolveTime(seamless)={t:.6f}s, SolveTime(mixed)={m:.6f}s".format(t=t_seamless, m=t_mixed))


if __name__ == "__main__":
    main()

