#!/bin/bash
# AWS GPU Launch Script for Critical Experiments #1 and #2
# Total: 9 experiments (3 h=24 + 6 LoRA rank)
# Estimated time: 6-8 hours wallclock with parallel execution

set -e

# ============================================================
# CRITICAL #1: Shorter Horizon (h=24) for Positive R² Floor
# Goal: Find (encoder, probe head) with R²(PT) > 0.2 on trained head AND ΔR² > 0
# ============================================================

echo "Launching Critical #1: h=24 experiments (3 seeds: 42, 101, 123)"

# Seed 42
aws batch submit-job \
  --job-name moirai-h24-seed42 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "B",
      "--horizon", "24",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "10000",
      "--seed", "42",
      "--results-dir", "results/v32_h24_positive_floor",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_h24_s42.json

# Seed 101
aws batch submit-job \
  --job-name moirai-h24-seed101 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "B",
      "--horizon", "24",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "10000",
      "--seed", "101",
      "--results-dir", "results/v32_h24_positive_floor",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_h24_s101.json

# Seed 123
aws batch submit-job \
  --job-name moirai-h24-seed123 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "B",
      "--horizon", "24",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "10000",
      "--seed", "123",
      "--results-dir", "results/v32_h24_positive_floor",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_h24_s123.json

echo "Critical #1 jobs submitted. Job IDs:"
jq -r '.jobId' /tmp/job_h24_s42.json
jq -r '.jobId' /tmp/job_h24_s101.json
jq -r '.jobId' /tmp/job_h24_s123.json

# ============================================================
# CRITICAL #2: LoRA-Large Rank Escalation 3-Seed Replication
# Goal: Strengthen "rank does not rescue" with 3 seeds per rank
# Current: seed 42 only; Adding: seeds 101, 123 for r=16,32,64
# ============================================================

echo ""
echo "Launching Critical #2: LoRA-Large rank escalation (6 experiments)"

# r=16, seed 101
aws batch submit-job \
  --job-name moirai-lora-r16-s101 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "16",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "101",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r16_s101.json

# r=16, seed 123
aws batch submit-job \
  --job-name moirai-lora-r16-s123 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "16",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "123",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r16_s123.json

# r=32, seed 101
aws batch submit-job \
  --job-name moirai-lora-r32-s101 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "32",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "101",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r32_s101.json

# r=32, seed 123
aws batch submit-job \
  --job-name moirai-lora-r32-s123 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "32",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "123",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r32_s123.json

# r=64, seed 101
aws batch submit-job \
  --job-name moirai-lora-r64-s101 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "64",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "101",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r64_s101.json

# r=64, seed 123
aws batch submit-job \
  --job-name moirai-lora-r64-s123 \
  --job-queue research-gpu-queue \
  --job-definition moirai-finetune:latest \
  --container-overrides '{
    "command": [
      "python", "scripts/finetune_forecasting.py",
      "--condition", "E",
      "--lora-rank", "64",
      "--model-size", "large",
      "--horizon", "96",
      "--data-path", "data/forecasting/ETTh2.csv",
      "--max-train-samples", "500",
      "--seed", "123",
      "--results-dir", "results/v32_lora_large_rank",
      "--device", "cuda"
    ]
  }' \
  --output json > /tmp/job_lora_r64_s123.json

echo "Critical #2 jobs submitted. Job IDs:"
jq -r '.jobId' /tmp/job_lora_r16_s101.json
jq -r '.jobId' /tmp/job_lora_r16_s123.json
jq -r '.jobId' /tmp/job_lora_r32_s101.json
jq -r '.jobId' /tmp/job_lora_r32_s123.json
jq -r '.jobId' /tmp/job_lora_r64_s101.json
jq -r '.jobId' /tmp/job_lora_r64_s123.json

echo ""
echo "All 9 experiments submitted to AWS Batch."
echo "Monitor status with: aws batch describe-jobs --jobs <job-id>"
echo "Expected completion: 6-8 hours"
