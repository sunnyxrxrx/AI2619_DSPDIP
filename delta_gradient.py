import os
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_channel_gradient_changes(src_path: str, mask_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取并归一化图像（转为 RGB 格式）
    src_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    mask_bool = mask > 0.5

    h, w, c = src.shape

    # 2. 计算 1.8 倍放大并截断后的图
    src_scaled = np.clip(src * 1.8, 0.0, 1.0)

    # 创建一个 3行 x 3列 的大画布
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=200)
    channel_names = ["Red (Channel 0)", "Green (Channel 1)", "Blue (Channel 2)"]

    for ch in range(3):
        # 3. 提取单通道图像
        s_ch = src[..., ch]
        s_scaled_ch = src_scaled[..., ch]

        # 4. 计算单通道原图梯度
        gx_orig = np.zeros_like(s_ch)
        gy_orig = np.zeros_like(s_ch)
        gx_orig[:, :-1] = s_ch[:, 1:] - s_ch[:, :-1]
        gy_orig[:-1, :] = s_ch[1:h, :] - s_ch[:-1, :]
        grad_orig = np.abs(gx_orig) + np.abs(gy_orig)

        # 5. 计算单通道截断图梯度
        gx_scaled = np.zeros_like(s_scaled_ch)
        gy_scaled = np.zeros_like(s_scaled_ch)
        gx_scaled[:, :-1] = s_scaled_ch[:, 1:] - s_scaled_ch[:, :-1]
        gy_scaled[:-1, :] = s_scaled_ch[1:h, :] - s_scaled_ch[:-1, :]
        grad_scaled = np.abs(gx_scaled) + np.abs(gy_scaled)

        # 6. 计算单通道梯度差值 (仅保留 Mask 内)
        grad_diff = np.zeros_like(grad_orig)
        grad_diff[mask_bool] = grad_scaled[mask_bool] - grad_orig[mask_bool]

        # 计算对称色彩范围
        max_abs_diff = max(np.max(np.abs(grad_diff)), 1e-12)

        # 7. 绘制该通道的三张子图
        # 列 1：原图梯度 (plasma 暖色图)
        im1 = axes[ch, 0].imshow(grad_orig, cmap="plasma", vmin=0.0, vmax=0.15)
        axes[ch, 0].set_title(
            f"{channel_names[ch]}: Original $|\\nabla S|$", fontsize=10
        )
        axes[ch, 0].axis("off")
        fig.colorbar(im1, ax=axes[ch, 0], fraction=0.046, pad=0.04)

        # 列 2：截断图梯度 (plasma 暖色图)
        im2 = axes[ch, 1].imshow(grad_scaled, cmap="plasma", vmin=0.0, vmax=0.15)
        axes[ch, 1].set_title(
            f"{channel_names[ch]}: $|\\nabla S|$ after Scaling and Clamping", fontsize=10
        )
        axes[ch, 1].axis("off")
        fig.colorbar(im2, ax=axes[ch, 1], fraction=0.046, pad=0.04)

        # 列 3：梯度变化差分图 (seismic 双极发散色图，正红负蓝)
        im3 = axes[ch, 2].imshow(
            grad_diff, cmap="seismic", vmin=-max_abs_diff, vmax=max_abs_diff
        )
        axes[ch, 2].set_title(
            f"{channel_names[ch]}: Change of $|\\nabla S|$", fontsize=10
        )
        axes[ch, 2].axis("off")
        cbar = fig.colorbar(im3, ax=axes[ch, 2], fraction=0.046, pad=0.04)
        if ch == 1:
            cbar.set_label(
                "Strengthened (+) vs. Weakened (-)", rotation=270, labelpad=15
            )

    fig.tight_layout()
    output_path = os.path.join(output_dir, "channel_gradient_changes.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# 运行测试
visualize_channel_gradient_changes(
    "figs/poisson_source.png",
    "figs/poisson_mask.png",
    "results/results_toy_reconstruction",
)