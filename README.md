# Trading and Allocation in PPO Portfolios

This repository provides a reproducible workflow for attributing reinforcement-learning portfolio performance to initial allocation and subsequent trading. A complementary exposure-composition decomposition separates persistent stock selection, exposure timing, and changes in stock composition. The code includes data preparation, PPO training, counterfactual replays, and six Jupyter notebooks. Data, models, and experiment records are generated and retained locally; they are not bundled in the Git repository.

## Study design

The complete study comprises 24 PPO models, each trained for 5,001,216 transitions. Every condition uses three random seeds:

| Condition | Policies |
| --- | ---: |
| Synthetic market with a strong or zero predictive signal | 6 |
| Single-stock trading in MSFT, JPM, or JNJ with cash | 9 |
| Joint allocation across MSFT, JPM, JNJ, and cash | 3 |
| Joint allocation across the fixed Dow 30 and cash | 3 |
| Joint allocation across 100 liquidity-selected stocks and cash | 3 |

The joint comparisons use 3-, 30-, and 100-stock universes with the same features, PPO settings, execution convention, and controls. The base task planner covers the original 18 synthetic, single-stock, and three-stock tasks. The Dow 30 and 100-stock extensions have separate preparation, training, and report entry points; adding them does not overwrite earlier results. Comparisons across universes are descriptive, not isolated causal effects of stock count.

Real-market training uses 2009–2018, with 2019 reserved for validation. The primary evaluation is 2020–2025; a continuous extension runs through 2026-08-31 without resetting holdings. The reference calendar spans 2008-03-19 through 2026-08-31. The Dow constituent list is fixed as of 2026-08-31 and applied retrospectively, so it is not a historical investable reconstruction of the index.

The 100-stock universe is selected from S&P 500 candidates cross-checked against IVV equity holdings. Eligibility requires pretraining history and at least 98% training-period coverage. One share class is retained per issuer, and eligible issuers are ranked by median raw Close × Volume over the final 252 training sessions ending in 2018. Neither validation nor test returns enter selection. This custom universe is not the S&P 100. Current candidate membership is retrospective and remains subject to survivorship bias. IVV is used only to verify membership, not as a traded asset or return benchmark.

All policies use a long-only, unleveraged target-weight environment with cash as an asset. Actions are chosen after the close and executed at the next open. The principal one-way transaction cost is 10 basis points. Evaluations preserve observations, actions, target and executed weights, rewards, fees, wealth, and counterfactual paths. First-allocation hold and Freeze-K retain actual units without further trading. Mean-exposure/composition replays are retrospective attribution tools, not deployable benchmarks. The chronological and exposure-composition decompositions use different references and should not be equated component by component.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_data_preparation_and_validation.ipynb` | Build or inspect raw, interim, processed, and frozen data. |
| `01_synthetic_signal_validation.ipynb` | Check learning behavior under known and absent signals. |
| `02_single_asset_timing.ipynb` | Evaluate the three single-stock policies. |
| `03_multi_asset_attribution.ipynb` | Evaluate the matched three-stock joint policy and counterfactuals. |
| `04_dow30_multi_asset_attribution.ipynb` | Evaluate the separate 30-stock joint-policy extension. |
| `05_liquid100_multi_asset_attribution.ipynb` | Evaluate the training-selected 100-stock joint-policy extension. |

Notebooks 01–05 default to `RUN_EXPERIMENTS = False`. Set the switch to `True` only when intentionally launching or resuming training; verified completed runs are skipped. Notebook 00 also disables downloads and preparation by default. Prepare the Dow data first. For the 100-stock experiment, enable its separate `PREPARE_LIQUID100` section in Notebook 00 before running Notebook 05. Preparation retains `liquid100_` files alongside the existing data layers and does not replace Dow data.

Kernel name: `dtasrl-py313`. Keep at most two formal training workers active and at least 8 GiB of available memory. Each extension runs its three seeds sequentially. Opening a notebook is not sufficient to supply missing data or results; the preparation and training steps must be explicitly enabled when required.

## Installation

Create the environment from the project root:

```bash
conda env create -f environment.yml
conda activate dtasrl-py313
```

The project targets Python 3.13 and uses Stable Baselines Jax (SBX) for PPO. Dependency versions are recorded in `uv.lock`. The training backend uses JAX on CPU; Stable-Baselines3 supplies shared environment, logging, and callback interfaces.

## Command-line workflow

Run commands from the project root. Data download requires network access; model evaluation and report generation use the frozen local inputs.

```bash
# Rebuild the market-data layers when a new snapshot is intentionally required.
dtasrl data build --config configs/data/dow30.yaml

# Freeze the training-period scalers and inputs for the primary study.
dtasrl data prepare

# Inspect the fixed primary 18-task plan without starting training.
dtasrl study plan

# Run one primary model. Use --resume only for a matching interrupted run.
dtasrl experiment run --config configs/experiments/synthetic_high.yaml --seed 101

# Summarize complete, identity-checked primary runs.
dtasrl study summarize
```

The extensions are not included in the base `dtasrl study plan` or `dtasrl study summarize`. Use Notebook 00 for preparation and Notebooks 04–05 for their training and reports. Their reusable modules are `scripts/dow30_extension.py` and `scripts/liquid100_extension.py`. Once the corresponding frozen inputs exist, an individual extension model can also be launched explicitly:

```bash
dtasrl experiment run --config configs/experiments/multi_dow30.yaml --seed 101
dtasrl experiment run --config configs/experiments/multi_liquid100.yaml --seed 101
```

The displayed seed labels 1–3 correspond to random seeds 101, 202, and 303. Every run retains the actual random seed. Downloading a new candidate list may change membership and data hashes; reproducing an existing result requires its original frozen inputs and provenance, not merely a fresh download from the same URL.

## Data and results

Market data are stored under `data/raw`, `data/interim`, and `data/processed`. Frozen experiment arrays and training-only scalers are under `data/processed/inputs`. The experiment outputs are separated by condition:

```text
outputs/
  01_synthetic/
  02_single_asset/
  03_multi_asset/
  04_dow30/
  05_liquid100/
  study_plan/
```

Each completed run retains its configuration and provenance, model and checkpoints, rollout-level training statistics, complete validation and test trajectories, action and observation records, counterfactual paths, CSV and Parquet tables, PNG figures, and SHA-256 inventories. The figures are derived from saved tables and can be regenerated without retraining. Research figures are exported at 320 DPI.

Yahoo Finance is the source of the market panel. Before redistributing downloaded market data or model outputs derived from it, verify the applicable data-use terms. This repository does not provide investment advice.

## Public copy

Tracked files contain the six notebooks, executable study code, data and experiment configurations, and dependency specifications. Executed notebook outputs and workstation metadata are removed. Market data, trained models, run logs, large result trees, manuscript drafts, internal planning notes, debugging utilities, and local test files are excluded. Consequently, cloning this repository does not download the 24 completed models or their research evidence. Results generated locally remain in the experiment-specific output folders and can be inspected without retraining. Before publishing data or derived outputs separately, verify redistribution rights and remove private metadata.
