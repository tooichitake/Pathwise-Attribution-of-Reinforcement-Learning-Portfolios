# Local data layout

The project keeps the standard `raw / interim / processed` layers without an additional dataset-name or timestamp directory.

```text
data/
  raw/
    market_data.csv
    market_data.parquet
    metadata.json
  interim/
    features_unscaled.csv
    features_unscaled.parquet
    metadata.json
  processed/
    panel.csv
    panel.parquet
    metadata.json
    inputs/
      real/
      synthetic_high/
      synthetic_zero/
```

The three top-level layers are reproducibility stages, not backups. CSV is the human-readable copy and Parquet is the type-preserving training copy; metadata records their equality, provenance, schema, and SHA-256 hashes.

The `inputs` directories contain different model inputs, not historical versions: real train/validation/test arrays and two distinct synthetic conditions. Active data never uses a timestamp or content hash as a directory name.
