# Trading and Allocation in PPO Portfolios

This repository studies whether the outcomes of a reinforcement-learning portfolio policy are associated with exposure timing, persistent stock composition, or subsequent trading. It provides the market-data pipeline, frozen experiment inputs, PPO training code, counterfactual replays, Jupyter notebooks, and locally saved results.

## Study design

The primary study consists of 18 PPO runs, each trained for 5,001,216 transitions with three prespecified random seeds:

| Condition | Policies |
| --- | ---: |
| Synthetic market with a strong or zero predictive signal | 6 |
| Single-stock trading in MSFT, JPM, or JNJ with cash | 9 |
| Joint allocation across MSFT, JPM, JNJ, and cash | 3 |

A separate extension trains three joint policies over all 30 fixed Dow constituents and cash. It does not change the primary 18-run analysis. The extension compares policies within the 30-stock universe; it does not treat a difference from the three-stock policy as an isolated effect of universe size.

The real-data panel spans the first date on which all 30 selected stocks have valid data through 2026-08-31. The constituent list is fixed as of 2026-08-31 and applied retrospectively. It is therefore a controlled, survivorship-affected universe, not a historical investable reconstruction of the Dow.

All policies use the same long-only, unleveraged target-weight execution convention with cash as an asset and transaction costs charged on traded value. The evaluation preserves daily observations, target and executed weights, rewards, costs, wealth, and counterfactual paths. Freeze-K and exposure-composition replays distinguish the initial allocation from subsequent adjustments. The test-period mean-composition paths are retrospective attribution tools, not deployable benchmarks.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_data_preparation_and_validation.ipynb` | Build or inspect raw, interim, processed, and frozen data. |
| `01_synthetic_signal_validation.ipynb` | Check learning behavior under known and absent signals. |
| `02_single_asset_timing.ipynb` | Evaluate the three single-stock policies. |
| `03_multi_asset_attribution.ipynb` | Evaluate the matched three-stock joint policy and counterfactuals. |
| `04_dow30_multi_asset_attribution.ipynb` | Evaluate the separate 30-stock joint-policy extension. |

Notebooks 01–04 default to `RUN_EXPERIMENTS = False`, so reopening them does not start training. Set the switch to `True` only when intentionally launching or resuming PPO jobs. Completed runs are skipped. The data notebook does not download new market data unless its rebuild switch is enabled.

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

The 30-stock extension has its own configuration, frozen inputs, notebook, and output directory. Its preparation and reporting code is in `scripts/dow30_extension.py`; it is not included in `dtasrl study summarize`.

## Data and results

Market data are stored under `data/raw`, `data/interim`, and `data/processed`. Frozen experiment arrays and training-only scalers are under `data/processed/inputs`. The experiment outputs are separated by condition:

```text
outputs/
  01_synthetic/
  02_single_asset/
  03_multi_asset/
  04_dow30/
  study_plan/
```

Each completed run retains its configuration and provenance, model and checkpoints, rollout-level training statistics, complete validation and test trajectories, action and observation records, counterfactual paths, CSV and Parquet tables, PNG figures, and SHA-256 inventories. The figures are derived from saved tables and can be regenerated without retraining. Research figures are exported at 320 DPI.

Yahoo Finance is the source of the market panel. Before redistributing downloaded market data or model outputs derived from it, verify the applicable data-use terms. This repository does not provide investment advice.

## Public copy

This copy keeps the five study notebooks, reproducible code and configuration, local data layers, and the saved outputs for all 21 completed runs. Draft manuscripts, internal planning notes, old research artifacts, debug utilities, and local test files are excluded. Large `data/` and `outputs/` trees are ignored by Git by default; review their size and redistribution rights before publishing them separately. The saved source snapshots and manifests identify the original training code and data, while portable paths in the public reports avoid relying on one workstation's directory layout.
