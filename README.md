# Poisson Image Editing
Final project of SJTU 2026 Spring AI2619: Digital Signal and Image Processing.

This project implements classic [Poisson image editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf), including seamless cloning and mixed-gradient blending.

The Poisson system construction and solving are unified in [poisson_cloner.py](https://github.com/sunnyxrxrx/AI2619_DSPDIP/blob/main/poisson_cloner.py) via the `PoissonCloner` class.

## Features

- Toy reconstruction (solve a Poisson system from gradients on a grayscale image)
- Seamless cloning: source-gradient Poisson blending + zoomed comparisons
- Mixed gradients: mixed-gradient Poisson blending + global/zoom comparisons
- Failure cases: brightness scaling and mask dilation studies + comparisons
- Runtime summaries: scripts write `runtime_summary.txt` with `N`, `nnz(A)`, `build_time_s`, `solve_time_s` (when applicable)

## Setup

Create and activate a conda environment, then install dependencies:

```bash
conda create -n poisson-edit python=3.10
conda activate poisson-edit
pip install -r requirements.txt
```

## Run

Start from the toy example:

```bash
python toy_reconstruction.py
```

Then run the other tasks:

```bash
python seamless_cloning.py
python mixed_gradients.py
python poisson_failure_cases.py
```

Outputs are saved under the corresponding `results/`.

