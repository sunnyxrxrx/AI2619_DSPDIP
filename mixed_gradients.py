import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import save_global_comparison, save_zoom_comparison, naive_copy_paste, _choose_zoom_centers, _write_text
from poison_cloner import PoissonCloner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="figs/poisson_source.png")
    parser.add_argument("--target_path", type=str, default="figs/poisson_target.png")
    parser.add_argument("--mask_path", type=str, default="figs/poisson_mask.png")
    parser.add_argument("--output_dir", type=str, default="results/results_task3")
    parser.add_argument("--zoom_half_size", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cloner = PoissonCloner(args.src_path, args.target_path, args.mask_path)
    # 【给分点B.4：copy-paste对比】
    naive = naive_copy_paste(cloner.src, cloner.target, cloner.mask)
    plt.imsave(os.path.join(args.output_dir, "naive_copy_paste.png"), naive)
    # 【给分点C.3:与 source-gradient 对比，结合报告】
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

    # 【给分点B.5：边界局部放大】
    # 【给分点C.4：局部细节分析，结合报告】
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
    # 【给分点：运行时间和矩阵信息】
    print(
        "N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}".format(
            n=info.n_unknowns, nnz=info.nnz, bt=info.build_time_s
        )
    )
    print("SolveTime(seamless)={t:.6f}s, SolveTime(mixed)={m:.6f}s".format(t=t_seamless, m=t_mixed))

    seamless_info = {
        "N": info.n_unknowns,
        "nnzA": info.nnz,
        "build_time_s": info.build_time_s,
        "solve_time_s": t_seamless,
    }
    mixed_info = {
        "N": info.n_unknowns,
        "nnzA": info.nnz,
        "build_time_s": info.build_time_s,
        "solve_time_s": t_mixed,
    }
    summary = []
    summary.append(
        "Seamless(source-gradient): N={N}, nnz(A)={nnzA}, build_time_s={build_time_s:.6f}, solve_time_s={solve_time_s:.6f}".format(
            **seamless_info
        )
    )
    summary.append(
        "Mixed-gradients: N={N}, nnz(A)={nnzA}, build_time_s={build_time_s:.6f}, solve_time_s={solve_time_s:.6f}".format(
            **mixed_info
        )
    )
    _write_text(os.path.join(args.output_dir, "runtime_summary.txt"), "\n".join(summary) + "\n", append=True)


if __name__ == "__main__":
    main()
