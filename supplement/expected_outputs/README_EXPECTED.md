# Expected Outputs

These CSVs contain the exact values reported in the paper, generated from our
10-seed experimental runs on an NVIDIA A10G GPU.

## Files

- **table2_etth2_sample_sweep.csv**: Sample-size sweep on ETTh2 (condition B,
  h=96). Shows how forgetting scales with training set size.

- **table3_task_native_probe.csv**: Task-native linear-forecaster probe R^2
  before and after fine-tuning. Delta R^2 > 0 indicates that fine-tuning
  restructured representations in a task-beneficial direction.

- **table4_diagnostic_summary.csv**: Cross-condition diagnostic summary with
  verdicts (harmful_drift, beneficial_restructuring, stable) derived from
  forgetting percentage thresholds.

- **table5_drift_diagnosis.csv**: Per-seed drift diagnosis combining CKA
  similarity with probe delta R^2 to disambiguate harmful forgetting from
  beneficial restructuring.

## Reproduction

To regenerate these from your own runs:

```bash
python scripts/06_make_tables.py --input runs/ --output expected_outputs/
```

Then compare against the provided CSVs. Small floating-point differences
(< 1e-4) are expected across GPU architectures due to non-associative
floating-point operations in cuDNN/cuBLAS.

## Matching to Paper Tables

- table2 -> Paper Table 2 (Sample-size sweep)
- table3 -> Paper Table: R^2_task columns
- table4 -> Paper Table 4 (Cross-domain diagnostic summary)
- table5 -> Paper Table 5 (Drift diagnosis)
