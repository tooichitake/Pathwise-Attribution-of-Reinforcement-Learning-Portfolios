# Local data layout

The project keeps the standard `raw / interim / processed` layers without an additional dataset-name or timestamp directory.

```text
data/
  raw/
    market_data.csv
    market_data.parquet
    metadata.json
    liquid100_*_source.csv
    liquid100_constituents.csv
    liquid100_constituents.parquet
    liquid100_candidate_market_data.csv
    liquid100_candidate_market_data.parquet
    liquid100_metadata.json
  interim/
    features_unscaled.csv
    features_unscaled.parquet
    metadata.json
    liquid100_features_unscaled.csv
    liquid100_features_unscaled.parquet
    liquid100_selection_audit.csv
    liquid100_selection_audit.parquet
  processed/
    panel.csv
    panel.parquet
    metadata.json
    liquid100_panel.csv
    liquid100_panel.parquet
    liquid100_metadata.json
    inputs/
      real/
      dow30/
      liquid100/
      synthetic_high/
      synthetic_zero/
```

The three top-level layers are reproducibility stages, not backups. CSV is the human-readable copy and Parquet is the type-preserving training copy; metadata records their equality, provenance, schema, and SHA-256 hashes.

The `inputs` directories contain different model inputs, not historical versions: the three-stock, Dow 30, and 100-stock train/validation/test arrays and two distinct synthetic conditions. Active data never uses a timestamp or content hash as a directory name.

The 100-stock files use a flat `liquid100_` prefix to distinguish a separate stock universe without replacing the Dow files. Its candidate crosscheck, eligibility decisions, selected issuers, and training-period liquidity ranking are retained in the selection audit. Frozen inputs and the training-only scaler are stored under `processed/inputs/liquid100`. The selected portfolio is not the S&P 100 index.

These data layers are generated locally and ignored by Git. The published repository contains preparation code and configuration, not downloaded Yahoo prices, private run metadata, or trained models. A newly retrieved candidate list can differ from the original frozen list; exact research reproduction requires the recorded snapshot and matching hashes.
