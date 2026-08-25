# StateFlowDiff

**StateFlowDiff: A Diffusion-Based Generative Framework With Traffic-State-Aware Aggregation for Holiday Traffic Flow Forecasting**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

StateFlowDiff is a diffusion-based generative forecasting framework designed for **holiday and other non-recurrent traffic conditions with limited historical coverage**. Instead of directly mapping one historical window to a single deterministic future, StateFlowDiff first generates multiple plausible microscopic traffic-flow realizations and then converts them into a **state-representative macroscopic prediction**.

The framework explicitly addresses three challenges in holiday traffic forecasting:

1. **Structured nonstationarity** caused by demand-level shifts, peak-phase changes, and short-term disturbances.
2. **Conditional future uncertainty**, where similar observed traffic states may correspond to multiple plausible future evolutions.
3. **Traffic-state consistency**, requiring generated trajectories to remain compatible with spatial propagation patterns and observed operating states.

> This repository provides the implementation, configurations, datasets, experimental outputs, and plotting utilities used for StateFlowDiff.

---

## Overview

StateFlowDiff consists of five main components organized into three stages:

### 1. Multi-scale variation representation

**Localized Spectral-Temporal Decoupling Embedding (LSTDE)** decomposes historical traffic flow into multiple frequency bands and learns localized temporal representations from band-specific patches. It is designed to capture traffic variations at different temporal scales, including slowly varying demand levels, peak-phase changes, and short-term fluctuations.

### 2. Microscopic traffic-flow realization generation

**Spatial Field Coupled Normalization (SFCN)** combines node-wise normalization with neighborhood relative-flow information to construct a spatial-field diffusion space.

A **Conditional Diffusion Module** then performs reverse denoising conditioned on historical observations, temporal information, and multi-scale representations to generate multiple plausible future traffic-flow realizations.

**Hybrid Trend-Residual Correction (HTRC)** further compensates for conservative peak growth and local transition errors by combining time-domain and frequency-domain trend information.

### 3. Macroscopic traffic-flow estimation

**Density-Centroid Aggregation (DCA)** assigns realization-specific weights according to sample-cloud density and historical-state compatibility. The weighted centroid is used as the final macroscopic traffic-flow prediction.

Conceptually,

```text
Historical Traffic Flow
        |
        v
+-------------------------------+
| LSTDE                         |
| Localized spectral-temporal   |
| multi-scale representation    |
+-------------------------------+
        |
        v
+-------------------------------+
| SFCN                          |
| Spatial-field normalization   |
+-------------------------------+
        |
        v
+-------------------------------+
| Conditional Diffusion         |
| Multiple future realizations  |
+-------------------------------+
        |
        v
+-------------------------------+
| HTRC                          |
| Trend-residual correction     |
+-------------------------------+
        |
        v
+-------------------------------+
| DCA                           |
| State-aware aggregation       |
+-------------------------------+
        |
        v
Macroscopic Traffic-Flow Forecast
```

---

## Main Features

- **Generative traffic forecasting** rather than single-path deterministic prediction.
- **Multi-scale spectral-temporal modeling** for non-recurrent traffic variations.
- **Graph-aware spatial-field normalization** for network-consistent generation.
- **Fast diffusion inference** using DPM-Solver++.
- **Multi-realization forecasting** for representing conditional future ambiguity.
- **Traffic-state-aware aggregation** instead of conventional mean or median aggregation.
- **Config-driven experiments** for multiple forecasting horizons and ablation studies.
- **Historical-coverage analysis** for studying when generative forecasting is advantageous over deterministic forecasting.

---

## Experimental Highlights

The manuscript evaluates StateFlowDiff primarily on **Fujian-30** for holiday traffic forecasting and uses **PeMS03, PeMS04, and PeMS08** to analyze performance under different levels of continuous historical coverage.

### Fujian-30 forecasting performance

| Model | H=12 MAE | H=24 MAE | H=36 MAE |
| --- | ---: | ---: | ---: |
| DLinear | 0.24138 | 0.25067 | 0.25466 |
| PatchTST | 0.23135 | 0.24074 | 0.24564 |
| iTransformer | 0.22770 | 0.23852 | 0.24521 |
| Diffusion-TS | 0.23353 | 0.24235 | 0.24823 |
| TSDiff | 0.23855 | 0.24914 | 0.25619 |
| SimDiff | 0.22740 | 0.23673 | 0.24201 |
| **StateFlowDiff** | **0.22411** | **0.23649** | **0.23866** |

StateFlowDiff achieves the lowest MAE, MSE, and RMSE across all three forecasting horizons reported in the manuscript. Its advantage becomes more pronounced at the longest forecasting horizon, where modeling future ambiguity and traffic-state transitions becomes increasingly important.

The manuscript additionally evaluates:

- holiday trend fidelity using **Trend Alignment (TA)** and **Step-Sign Consistency (SSC)**;
- module ablations for **LSTDE**, **SFCN**, and **HTRC**;
- frequency-band contributions;
- aggregation strategies including Single Sample, Mean, Median, MoM, KDE Mode, and DCA;
- sampling-budget sensitivity;
- historical-coverage applicability on PeMS datasets.

---

## Repository Structure

```text
StateFlowDiff/
├── StateFlowDiff/
│   ├── data_provider/        # Data loading and preprocessing
│   ├── exp/                  # Experiment and forecasting pipelines
│   ├── frequency/            # Frequency-domain operations
│   ├── layers/               # Neural-network building blocks
│   ├── macro/                # Macroscopic aggregation components
│   ├── micro/                # Microscopic generative components
│   ├── models/               # StateFlowDiff model definitions
│   ├── utils/                # Utilities, metrics, logging, etc.
│   └── train.py              # Main training / evaluation entry point
├── configs/
│   └── fujian30/
│       ├── stateflowdiff.yaml
│       ├── stateflowdiff_h12.yaml
│       ├── stateflowdiff_h24.yaml
│       ├── stateflowdiff_h36.yaml
│       └── ablation/
├── datasets/
│   ├── fujian-30/            # Fujian-30 dataset
│   ├── original_pems/        # Original PeMS datasets
│   └── pems_r30/             # Fixed-size PeMS region splits
├── plots/                    # Plotting scripts and paper figures
├── results/                  # Experimental results
├── requirements.txt
├── LICENSE
└── README.md
```

Dataset-construction/splitting utilities and baseline implementations are not included in the current repository release.

---

## Requirements

The current implementation depends on:

```text
numpy
pandas
scipy
pyyaml
matplotlib
torch
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for diffusion-model training and multi-sample inference.

---

## Data

### Fujian-30

Fujian-30 contains traffic-flow observations from 30 spatially connected expressway sections in Ningde, Fujian Province, China.

The repository expects the dataset under:

```text
datasets/fujian-30/
```

The released Fujian-30 configurations use files such as:

```text
datasets/fujian-30/fujian30_clean.csv
datasets/fujian-30/adjacent_gantry.csv
```

The data configuration identifies:

- `time_slot` as the temporal field,
- `station_index` as the sensor identifier,
- `traffic_flow` as the forecasting target,
- `is_holiday` as the holiday indicator.

### PeMS datasets

PeMS03, PeMS04, and PeMS08 are used for the historical-coverage applicability analysis. The repository contains directories for the original PeMS data and the fixed-size 30-sensor regional subsets used in the repeated-region experiments.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/zouguojian/StateFlowDiff.git
cd StateFlowDiff
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate StateFlowDiff on Fujian-30 with a 12-step prediction horizon:

```bash
python -m StateFlowDiff.train \
    --config configs/fujian30/stateflowdiff_h12.yaml
```

For the other prediction horizons:

```bash
python -m StateFlowDiff.train \
    --config configs/fujian30/stateflowdiff_h24.yaml

python -m StateFlowDiff.train \
    --config configs/fujian30/stateflowdiff_h36.yaml
```

The YAML configuration files control dataset paths, forecasting horizons, model dimensions, diffusion settings, sampling budgets, aggregation strategies, random seeds, and output locations.

---

## Historical-Coverage Experiments

The training entry point supports continuous-history control through `--history_ratio`.

For example:

```bash
python -m StateFlowDiff.train \
    --config configs/fujian30/stateflowdiff_h12.yaml \
    --history_ratio 0.50
```

The implementation keeps the validation and test periods fixed while changing the amount of continuous training history. This supports the applicability analysis described in the paper, where generative and deterministic forecasting are compared under different historical-coverage levels.

The default chronological split follows a `7:1:2` train/validation/test protocol, and `history_ratio` must not exceed the training split ratio.

---

## Configuration

Representative configuration groups include:

```yaml
seq_len: 96
pred_len: 36

use_lstde: true
use_sfcn: true

num_bands: 4
patch_len: 12

diff_steps: 100
s_steps: 5
sample_times: 20

test_aggregation_mode: dca
dca_bandwidth_kde: 15.0
dca_bandwidth_hist: 20.0
```

For exact reproduction, please use the corresponding YAML file in `configs/`. The configuration file used for a run should be treated as the authoritative runtime specification.

---

## Ablation Studies

Ablation configurations are provided under:

```text
configs/fujian30/ablation/
```

The manuscript studies the effects of removing the major trainable components:

- **w/o LSTDE**: temporal modeling without spectral-temporal decoupling;
- **w/o SFCN**: spatial-field coupling removed, reducing the transformation to node-wise normalization;
- **w/o HTRC**: diffusion realizations used without trend-residual correction.

DCA is evaluated separately as an inference-time aggregation strategy because it does not alter the trained generator.

---

## Sampling and Aggregation

At inference, StateFlowDiff repeatedly samples the conditional diffusion model to obtain multiple candidate traffic-flow realizations.

The paper uses:

- `Q = 100` forward diffusion steps;
- five-step DPM-Solver++ inference;
- `K = 20` sampled realizations for the default macroscopic prediction.

The candidate realizations can be aggregated using different strategies. StateFlowDiff proposes **Density-Centroid Aggregation (DCA)**, which jointly considers:

1. local sample-cloud concentration; and
2. compatibility with historical traffic states.

This design is intended to avoid placing the final point estimate between distinct high-density future modes or over-weighting a nonrepresentative realization cluster.

---

## Reproducibility

The training runner explicitly sets random seeds for Python, NumPy, and PyTorch and enables deterministic cuDNN behavior when possible.

The manuscript reports experiments using random seeds:

```text
0, 42, 2021
```

For aggregation analysis, multiple initial-noise seeds are used to evaluate the stability of inference-time aggregation.

Because implementation details may evolve during repository cleanup, **please use the committed YAML configurations rather than manually reconstructing hyperparameters from the manuscript text**.

---

## Output

By default, configurations define output/checkpoint directories such as:

```text
outputs/checkpoints/fujian30
```

The repository also contains:

```text
results/
plots/
```

for storing experimental outputs and reproducing paper figures.

---

## Citation

If you find StateFlowDiff useful in your research, please cite the corresponding paper:

```bibtex
@article{zou2026stateflowdiff,
  title   = {StateFlowDiff: A Diffusion-Based Generative Framework With Traffic-State-Aware Aggregation for Holiday Traffic Flow Forecasting},
  author  = {Zou, Guojian and Yan, Yijin and Zhao, Zixi and Ding, Weiping and Li, Ye},
  year    = {2026},
  note    = {Manuscript}
}
```

The citation information will be updated after publication.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions about the code or the paper, please open a GitHub issue or contact:

**Guojian Zou**  
College of Information, Mechanical and Electrical Engineering, Shanghai Normal University  
Key Laboratory of Road and Traffic Engineering, Ministry of Education, Tongji University  
Email: `guojianzou@shnu.edu.cn`

---

## Acknowledgements

This repository accompanies the StateFlowDiff study on generative traffic-flow forecasting under limited historical coverage. We thank the authors and maintainers of the open-source time-series forecasting and diffusion-model libraries that support reproducible research in this area.
