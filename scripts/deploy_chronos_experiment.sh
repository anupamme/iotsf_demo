#!/bin/bash
# Deploy and launch Chronos experiments on both GPU instances.
#
# Prerequisites:
#   - Both GPU instances running with SSH access
#   - Key at /Users/mediratta/keys/nips_east_1.pem
#
# Usage:
#   bash scripts/deploy_chronos_experiment.sh GPU1_HOST GPU2_HOST
#
# Example:
#   bash scripts/deploy_chronos_experiment.sh \
#     ec2-X-X-X-X.compute-1.amazonaws.com \
#     ec2-Y-Y-Y-Y.compute-1.amazonaws.com

set -e

KEY="/Users/mediratta/keys/nips_east_1.pem"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
GPU1="${1:?Usage: $0 GPU1_HOST GPU2_HOST}"
GPU2="${2:?Usage: $0 GPU1_HOST GPU2_HOST}"
REMOTE_DIR="~/workspace/iotsf_demo"

echo "=========================================="
echo "Deploying Chronos experiment"
echo "  GPU1: $GPU1"
echo "  GPU2: $GPU2"
echo "=========================================="

# Files to sync
FILES_TO_SYNC=(
    "scripts/finetune_chronos_m4.py"
    "scripts/launch_chronos_gpu1.sh"
    "scripts/launch_chronos_gpu2.sh"
    "src/"
)

# Deploy to GPU1
echo ""
echo "--- Deploying to GPU1 ---"
for f in "${FILES_TO_SYNC[@]}"; do
    if [ -d "$f" ]; then
        scp $SSH_OPTS -r "$f" ubuntu@${GPU1}:${REMOTE_DIR}/"$f"
    else
        ssh $SSH_OPTS ubuntu@${GPU1} "mkdir -p ${REMOTE_DIR}/$(dirname $f)"
        scp $SSH_OPTS "$f" ubuntu@${GPU1}:${REMOTE_DIR}/"$f"
    fi
done
echo "  Done."

# Deploy to GPU2
echo ""
echo "--- Deploying to GPU2 ---"
for f in "${FILES_TO_SYNC[@]}"; do
    if [ -d "$f" ]; then
        scp $SSH_OPTS -r "$f" ubuntu@${GPU2}:${REMOTE_DIR}/"$f"
    else
        ssh $SSH_OPTS ubuntu@${GPU2} "mkdir -p ${REMOTE_DIR}/$(dirname $f)"
        scp $SSH_OPTS "$f" ubuntu@${GPU2}:${REMOTE_DIR}/"$f"
    fi
done
echo "  Done."

# Install dependencies (if needed)
echo ""
echo "--- Checking dependencies ---"
for HOST in $GPU1 $GPU2; do
    ssh $SSH_OPTS ubuntu@${HOST} "cd ${REMOTE_DIR} && pip install chronos-forecasting pandas scikit-learn loguru 2>/dev/null | tail -1"
done
echo "  Done."

# Launch experiments
echo ""
echo "--- Launching GPU1 (Condition B: n=500 + n=10k) ---"
ssh $SSH_OPTS ubuntu@${GPU1} "cd ${REMOTE_DIR} && chmod +x scripts/launch_chronos_gpu1.sh && nohup bash scripts/launch_chronos_gpu1.sh > logs/chronos_gpu1.log 2>&1 &"
echo "  Launched in background. Log: ${REMOTE_DIR}/logs/chronos_gpu1.log"

echo ""
echo "--- Launching GPU2 (Controls: frozen + random-init) ---"
ssh $SSH_OPTS ubuntu@${GPU2} "cd ${REMOTE_DIR} && chmod +x scripts/launch_chronos_gpu2.sh && nohup bash scripts/launch_chronos_gpu2.sh > logs/chronos_gpu2.log 2>&1 &"
echo "  Launched in background. Log: ${REMOTE_DIR}/logs/chronos_gpu2.log"

echo ""
echo "=========================================="
echo "Experiments launched!"
echo ""
echo "Monitor progress:"
echo "  GPU1: ssh $SSH_OPTS ubuntu@${GPU1} 'tail -f ${REMOTE_DIR}/logs/chronos_gpu1.log'"
echo "  GPU2: ssh $SSH_OPTS ubuntu@${GPU2} 'tail -f ${REMOTE_DIR}/logs/chronos_gpu2.log'"
echo ""
echo "Check completion:"
echo "  GPU1: ssh $SSH_OPTS ubuntu@${GPU1} 'ls ${REMOTE_DIR}/results/chronos_m4_n10k/seed999/'"
echo "  GPU2: ssh $SSH_OPTS ubuntu@${GPU2} 'ls ${REMOTE_DIR}/results/chronos_m4_randinit/seed999/'"
echo "=========================================="
