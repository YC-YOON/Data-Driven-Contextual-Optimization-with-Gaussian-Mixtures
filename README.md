# Data-Driven Contextual Optimization with Gaussian Mixtures

This repository contains the experimental code and data accompanying the paper:

> YoungChul Yoon, Grani A. Hanasusanto, and Yijie Wang.  
> **Data-Driven Contextual Optimization with Gaussian Mixtures: Flow-Based Generalization, Robust Models, and Multistage Extensions.**  
> arXiv:2509.14557, 2025. [Paper](https://arxiv.org/abs/2509.14557) · [PDF](https://arxiv.org/pdf/2509.14557)

## Overview

The paper develops a Gaussian-mixture-model (GMM) framework for contextual stochastic optimization. It models the joint distribution of uncertain outcomes and side information, uses the resulting conditional Gaussian mixture in downstream optimization, extends the framework to general distributions with normalizing flows, adds distributionally robust formulations, and develops a GMM-based approximation for multistage problems with Markovian uncertainty.

The repository implements the paper's three numerical studies:

1. **Inventory management / contextual newsvendor** on synthetic multimodal data, including the GMM, GMM with normalizing flows (GMM-NF), and benchmark methods.
2. **Contextual portfolio optimization** with a mean-CVaR objective and market side information.
3. **Multistage wind-energy planning** using SDDP-style methods for Ohio and North Carolina data.

## Repository structure

```text
.
├── data/                         # Processed inputs and experiment data
│   ├── wind_data/                # Wind-generation and price MAT files
│   └── sp500_cache/              # Cached financial-market inputs
└── src/
    ├── Newsvendor.py             # CVXPY newsvendor optimization model
    ├── NV_LinQuad_s1.ipynb       # Inventory experiment, context dimension 1
    ├── NV_LinQuad_s5.ipynb       # Inventory experiment, context dimension 5
    ├── NV_LinQuad_s20.ipynb      # Inventory experiment, context dimension 20
    ├── NV_running_time.ipynb     # Inventory runtime experiment
    ├── NV_running_time_earlystopping.ipynb
    ├── NV_evaluation.ipynb       # Inventory result evaluation
    ├── PF_data_processing.ipynb  # Portfolio data preparation
    ├── PF_CVaR-Mean.ipynb        # Contextual mean-CVaR experiment
    ├── PF_evaluation.ipynb       # Portfolio result evaluation
    ├── WE_data_processing.ipynb  # Wind-data preparation
    ├── WE_SDDP_OH.ipynb          # Ohio multistage experiment
    ├── WE_SDDP_NC.ipynb          # North Carolina multistage experiment
    ├── WE_evaluation.ipynb       # Wind-energy result evaluation
    └── Results/                   # Saved experiment outputs
```

## Environment

The code is organized as Jupyter notebooks and uses Python 3.11. Choose either the Conda setup (recommended) or the standard `venv`/pip setup below.

### Option 1: Conda

Create the environment from `environment.yml` and activate it:

```bash
conda env create -f environment.yml
conda activate contextual-gmm
```

To update an existing environment after the dependency files change, run:

```bash
conda env update -f environment.yml --prune
```

### Option 2: Python venv and pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the virtual environment with `.venv\\Scripts\\activate` instead.

Some experiments also require licensed optimization software:

- `Newsvendor.py` uses **MOSEK** through CVXPY by default. A working MOSEK installation and license are required unless the solver is changed in the code.
- The wind-energy notebooks use **Gurobi** (`gurobipy`) and require a working Gurobi installation and license.

Install those packages according to the vendors' instructions and verify that their licenses are available before running the relevant notebooks.

## Running the experiments

Paths in the notebooks are relative to `src/`, so start Jupyter from that directory:

```bash
cd src
jupyter lab
```

Suggested execution order:

### Inventory management

Run one or more of:

```text
NV_LinQuad_s1.ipynb
NV_LinQuad_s5.ipynb
NV_LinQuad_s20.ipynb
NV_running_time.ipynb
NV_running_time_earlystopping.ipynb
```

Then use `NV_evaluation.ipynb` to summarize and visualize the saved results.

### Portfolio optimization

1. Run `PF_data_processing.ipynb` if the processed input data must be rebuilt.
2. Run `PF_CVaR-Mean.ipynb`.
3. Run `PF_evaluation.ipynb`.

Rebuilding the financial dataset may require internet access, a WRDS account, and permission to access the underlying WRDS data. Cached or processed files included under `data/` can be used where applicable.

### Multistage wind-energy planning

1. Run `WE_data_processing.ipynb` if the processed inputs must be rebuilt.
2. Run `WE_SDDP_OH.ipynb` and/or `WE_SDDP_NC.ipynb`.
3. Run `WE_evaluation.ipynb`.

The wind experiments require Gurobi and use the MAT files under `data/wind_data/`.

## Reproducibility notes

- The repository includes saved CSV outputs under `src/Results/` for evaluation.
- Full experiments, especially normalizing-flow training and multistage optimization, may be computationally intensive.
- Results can depend on package versions, solver versions, random seeds, hardware, and solver tolerances.
- Before rerunning data-processing notebooks, review all download credentials, date ranges, cache paths, and data-provider terms.

## Citation

If you use this repository, please cite:

```bibtex
@article{yoon2025datadriven,
  title   = {Data-Driven Contextual Optimization with Gaussian Mixtures: Flow-Based Generalization, Robust Models, and Multistage Extensions},
  author  = {Yoon, YoungChul and Hanasusanto, Grani A. and Wang, Yijie},
  journal = {arXiv preprint arXiv:2509.14557},
  year    = {2025},
  doi     = {10.48550/arXiv.2509.14557}
}
```

## Data and licensing

The source code in this repository is released under the [MIT License](LICENSE). Data files and data obtained from third-party services (including WRDS, Yahoo Finance, NLDAS, and PJM) may be governed by their respective providers' terms and are not relicensed by the MIT License unless explicitly stated.
