# AWS GPU Setup Instructions for Critical Experiments

## Prerequisites

1. **AWS Account** with Batch access
2. **IAM permissions** for batch:SubmitJob, batch:DescribeJobs
3. **ECR repository** with Docker image containing:
   - Python 3.10+
   - PyTorch 2.0+ with CUDA
   - uni2ts library (with PackedStdScaler patch)
   - Your code at `/workspace/iotsf_demo/`
   - Data at `/workspace/iotsf_demo/data/forecasting/ETTh2.csv`

## Setup Steps

### 1. Create AWS Batch Job Definition

```bash
# Create job definition JSON
cat > moirai-finetune-jobdef.json <<'EOF'
{
  "jobDefinitionName": "moirai-finetune",
  "type": "container",
  "containerProperties": {
    "image": "YOUR_ECR_REPO:moirai-finetune-latest",
    "vcpus": 4,
    "memory": 16384,
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ],
    "jobRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchJobRole",
    "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/BatchExecutionRole",
    "mountPoints": [
      {
        "sourceVolume": "data",
        "containerPath": "/workspace/iotsf_demo/data",
        "readOnly": false
      },
      {
        "sourceVolume": "results",
        "containerPath": "/workspace/iotsf_demo/results",
        "readOnly": false
      }
    ],
    "volumes": [
      {
        "name": "data",
        "host": {"sourcePath": "/mnt/efs/iotsf_demo/data"}
      },
      {
        "name": "results",
        "host": {"sourcePath": "/mnt/efs/iotsf_demo/results"}
      }
    ],
    "environment": [
      {"name": "PYTORCH_ENABLE_MPS_FALLBACK", "value": "1"},
      {"name": "CUDA_VISIBLE_DEVICES", "value": "0"}
    ]
  }
}
EOF

# Register job definition
aws batch register-job-definition --cli-input-json file://moirai-finetune-jobdef.json
```

### 2. Create Compute Environment (if not exists)

```bash
aws batch create-compute-environment \
  --compute-environment-name research-gpu-env \
  --type MANAGED \
  --compute-resources '{
    "type": "EC2",
    "minvCpus": 0,
    "maxvCpus": 64,
    "desiredvCpus": 0,
    "instanceTypes": ["p3.2xlarge", "g4dn.xlarge"],
    "subnets": ["subnet-xxxxx"],
    "securityGroupIds": ["sg-xxxxx"],
    "instanceRole": "arn:aws:iam::YOUR_ACCOUNT:instance-profile/BatchInstanceRole"
  }' \
  --service-role arn:aws:iam::YOUR_ACCOUNT:role/BatchServiceRole
```

### 3. Create Job Queue (if not exists)

```bash
aws batch create-job-queue \
  --job-queue-name research-gpu-queue \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=research-gpu-env
```

### 4. Build and Push Docker Image

```bash
cd /Users/mediratta/code/iotsf_demo

# Create Dockerfile if not exists
cat > Dockerfile.aws <<'EOF'
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY . /workspace/iotsf_demo/

# Install dependencies
RUN pip3 install -r /workspace/iotsf_demo/requirements.txt

# Apply uni2ts PackedStdScaler patch
RUN python3 /workspace/iotsf_demo/scripts/apply_scaler_patch.py

CMD ["bash"]
EOF

# Build and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -f Dockerfile.aws -t moirai-finetune .
docker tag moirai-finetune:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/moirai-finetune:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/moirai-finetune:latest
```

## Launch Experiments

```bash
# Make launch script executable
chmod +x aws_launch_critical_experiments.sh

# Launch all 9 experiments
./aws_launch_critical_experiments.sh
```

## Monitor Progress

```bash
# Get job IDs from launch output, then:
aws batch describe-jobs --jobs <job-id-1> <job-id-2> ... <job-id-9>

# Check specific job logs
aws logs tail /aws/batch/job --follow --log-stream-name <job-id>
```

## Retrieve Results

After jobs complete (6-8 hours):

```bash
# Download results from S3 or EFS
aws s3 sync s3://your-results-bucket/v32_h24_positive_floor/ results/v32_h24_positive_floor/
aws s3 sync s3://your-results-bucket/v32_lora_large_rank/ results/v32_lora_large_rank/

# Verify all JSONs present
ls results/v32_h24_positive_floor/*.json  # expect 3 files
ls results/v32_lora_large_rank/*.json     # expect 6 files
```

## Cost Estimate

- **Instance type**: p3.2xlarge (~$3/hour) or g4dn.xlarge (~$0.50/hour)
- **Per experiment**: 2-3 hours max
- **Total**: 9 experiments × 2.5 hours × $0.50 = ~$11-$14 (g4dn) or ~$67.50 (p3)
- **Recommendation**: Use g4dn.xlarge (sufficient for Moirai-Small/Large)

## Fallback: Run Locally in Sequence

If AWS setup is not feasible, run locally:

```bash
cd /Users/mediratta/code/iotsf_demo

# Critical #1: h=24 (3 experiments × 2-3 hours each)
for seed in 42 101 123; do
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_forecasting.py \
    --condition B --horizon 24 \
    --data-path data/forecasting/ETTh2.csv \
    --max-train-samples 10000 --seed $seed \
    --results-dir results/v32_h24_positive_floor --device mps
done

# Critical #2: LoRA rank (6 experiments × 4-5 hours each)
for rank in 16 32 64; do
  for seed in 101 123; do
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_forecasting.py \
      --condition E --lora-rank $rank --model-size large \
      --horizon 96 --data-path data/forecasting/ETTh2.csv \
      --max-train-samples 500 --seed $seed \
      --results-dir results/v32_lora_large_rank --device mps
  done
done
```

**Local total time**: ~6-9 hours (Critical #1) + ~24-30 hours (Critical #2) = **30-40 hours**
