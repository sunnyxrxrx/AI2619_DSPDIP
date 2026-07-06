import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple

def _read_grayscale_float64(img_path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img_f = img.astype(np.float64)
        if img_f.max() > 1.0:
            img_f /= 255.0
        return np.clip(img_f, 0.0, 1.0)
    except Exception:
        img = mpimg.imread(img_path)
        if img.ndim == 3:
            img = img[..., 0]
        img_f = img.astype(np.float64)
        if img_f.max() > 1.0:
            img_f /= 255.0
        return np.clip(img_f, 0.0, 1.0)
    

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
    '''
    Naive Copy-Paste（baseline）。

    将 source 在 mask 内的像素直接覆盖到 target 上，不进行任何梯度域/泊松融合。

    Parameters
    ----------
    src: (H, W, 3) float64
        源图（已归一化到 [0,1]）。
    target: (H, W, 3) float64
        目标图（已归一化到 [0,1]）。
    mask: (H, W) bool
        融合区域，True 表示来自源图。
    '''
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
    '''
    保存实验 C 的三种全局结果对比图（用于报告直接引用）。

    从左到右分别为：
    1) Naive Copy-Paste
    2) Source-gradient Poisson（实验 B）
    3) Mixed-gradient Poisson（实验 C）

    Parameters
    ----------
    naive_img, seamless_img, mixed_img: (H, W, 3) float64
        三种方法得到的结果图，像素范围应在 [0,1]。
    out_path: str
        输出路径（.png 等）。
    '''
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
    mixed_img: Optional[np.ndarray],
    center: Tuple[int, int],
    half_size: int,
    out_path: str,
    title: str,
) -> None:
    '''
    保存局部放大对比图：在同一块局部区域内对比两/三种方法的效果。

    说明：
    - center 表示局部区域中心 (cy, cx)
    - half_size 表示半边长，裁剪窗口为 [cy-half_size, cy+half_size) × [cx-half_size, cx+half_size)
    - 若 mixed_img 不为 None：三幅图按 (naive, seamless, mixed) 顺序横向排布
    - 若 mixed_img 为 None：两幅图按 (naive, seamless) 顺序横向排布

    Parameters
    ----------
    naive_img, seamless_img: (H, W, 3) float64
        两种方法的结果图。
    mixed_img: (H, W, 3) float64 | None
        Mixed-gradient Poisson 结果图；若为 None，则只绘制 naive 与 seamless 的两图对比。
    center: (cy, cx)
        局部区域中心坐标（像素坐标，y 在前）。
    half_size: int
        裁剪半边长。
    out_path: str
        输出路径。
    title: str
        图标题（用于报告描述局部区域用途）。
    '''
    h, w = naive_img.shape[:2]
    cy, cx = center
    y0, y1, x0, x1 = _crop_box_centered(h, w, cy, cx, half_size)

    if mixed_img is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
        imgs = (naive_img, seamless_img)
        titles = ("Naive Copy-Paste", "Source-gradient Poisson")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        imgs = (naive_img, seamless_img, mixed_img)
        titles = ("Naive Copy-Paste", "Source-gradient Poisson", "Mixed-gradient Poisson")

    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(np.clip(img[y0:y1, x0:x1, :], 0.0, 1.0))
        ax.set_title(t)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



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
    '''
    保存实验 B 的“边界局部放大对比图”。

    做法：
    - 使用 mask 的包围盒（外扩 margin 像素）作为局部放大区域
    - 左图为 naive copy-paste，右图为 poisson seamless cloning
    - 该图用于突出显示接缝（seam）消除效果

    Parameters
    ----------
    naive_img: (H, W, 3) float64
        直接粘贴结果。
    poisson_img: (H, W, 3) float64
        泊松无缝克隆结果。
    mask: (H, W) bool
        融合区域。
    out_path: str
        输出路径。
    margin: int
        包围盒向外扩展的像素数。
    '''
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

def _save_color(path: str, img: np.ndarray) -> None:
    '''
    保存 RGB 图像到磁盘（像素范围会被截断到 [0,1]）。
    '''
    plt.imsave(path, np.clip(img, 0.0, 1.0))


def _save_mask_png(path: str, mask: np.ndarray) -> None:
    '''
    保存二值 mask 到磁盘（以灰度图形式保存）。
    '''
    plt.imsave(path, mask.astype(np.float64), cmap="gray", vmin=0.0, vmax=1.0)


def _save_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    out_path: str,
) -> None:
    '''
    将两张结果图做左右对比并保存（用于 failure case 的全局对比）。

    Parameters
    ----------
    left, right: (H, W, 3) float64
        两张待对比图像。
    left_title, right_title: str
        左/右子图标题。
    out_path: str
        保存路径。
    '''
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].imshow(np.clip(left, 0.0, 1.0))
    axes[0].set_title(left_title)
    axes[0].axis("off")
    axes[1].imshow(np.clip(right, 0.0, 1.0))
    axes[1].set_title(right_title)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_crop_compare(
    img_a: np.ndarray,
    img_b: np.ndarray,
    box: Tuple[int, int, int, int],
    title_a: str,
    title_b: str,
    out_path: str,
    suptitle: str,
) -> None:
    '''
    对同一个局部裁剪区域做左右对比并保存（用于 failure case 的局部放大图）。

    Parameters
    ----------
    img_a, img_b: (H, W, 3) float64
        两张待对比图像。
    box: (y0, y1, x0, x1)
        裁剪区域。
    title_a, title_b: str
        左/右子图标题。
    out_path: str
        保存路径。
    suptitle: str
        总标题。
    '''
    y0, y1, x0, x1 = box
    crop_a = img_a[y0:y1, x0:x1, :]
    crop_b = img_b[y0:y1, x0:x1, :]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    axes[0].imshow(np.clip(crop_a, 0.0, 1.0))
    axes[0].set_title(title_a)
    axes[0].axis("off")
    axes[1].imshow(np.clip(crop_b, 0.0, 1.0))
    axes[1].set_title(title_b)
    axes[1].axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_text(path: str, content: str, append: bool = True) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(content)


@dataclass
class BuildInfo:
    n_unknowns: int
    nnz: int
    build_time_s: float


