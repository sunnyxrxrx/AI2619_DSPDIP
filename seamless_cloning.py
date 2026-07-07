import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import naive_copy_paste, save_zoom_comparison, _choose_zoom_centers, _write_text
from Possion.poisson_cloner import PoissonCloner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="figs/poisson_source.png")
    parser.add_argument("--target_path", type=str, default="figs/poisson_target.png")
    parser.add_argument("--mask_path", type=str, default="figs/poisson_mask.png")
    parser.add_argument("--output_dir", type=str, default="results/results_task2")
    parser.add_argument("--zoom_half_size", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cloner = PoissonCloner(args.src_path, args.target_path, args.mask_path)
    naive = naive_copy_paste(cloner.src, cloner.target, cloner.mask)
    plt.imsave(os.path.join(args.output_dir, "naive_copy_paste.png"), naive)

    poisson, info, solve_time_s = cloner.solve(mode="seamless")
    plt.imsave(os.path.join(args.output_dir, "poisson_seamless_cloning.png"), poisson)

    center1, center2 = _choose_zoom_centers(cloner.src, cloner.target, cloner.mask)
    save_zoom_comparison(
        naive,
        poisson,
        mixed_img=None,
        center=center1,
        half_size=args.zoom_half_size,
        out_path=os.path.join(args.output_dir, "zoom_area_1.png"),
        title="Zoom Area 1: Background Texture Penetration",
    )
    save_zoom_comparison(
        naive,
        poisson,
        mixed_img=None,
        center=center2,
        half_size=args.zoom_half_size,
        out_path=os.path.join(args.output_dir, "zoom_area_2.png"),
        title="Zoom Area 2: Boundary Transition",
    )

    print(
        "N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}, solve_time_s={st:.6f}".format(
            n=info.n_unknowns, nnz=info.nnz, bt=info.build_time_s, st=solve_time_s
        )
    )
    summary = []
    summary.append(
        "Seamless(source-gradient): N={n}, nnz(A)={nnz}, build_time_s={bt:.6f}, solve_time_s={st:.6f}".format(
            n=info.n_unknowns, nnz=info.nnz, bt=info.build_time_s, st=solve_time_s
        )
    )
    _write_text(os.path.join(args.output_dir, "runtime_summary.txt"), "\n".join(summary) + "\n", append=True)


if __name__ == "__main__":
    main()
