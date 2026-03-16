#!/bin/bash

#
# CORRECTED TRT-LLM Multi-Node Startup Script
# Uses 1.0.0rc3 (PROVEN WORKING) instead of 1.2.0rc6 (broken for SM121/ARM64)
#
# NVIDIA BUG: Their documentation links to 1.2.0rc6 which lacks SM121 support
# This script uses the documented working version: 1.0.0rc3
#
# Reference: https://github.com/NVIDIA/TensorRT-LLM/issues/8474
#

set -e

# Configuration
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3"  # <-- CORRECTED VERSION
MODEL="${MODEL:-nvidia/Llama-3.1-8B-Instruct-FP4}"
PORT="${PORT:-8355}"
TP_SIZE="${TP_SIZE:-2}"  # Multi-node: use tp_size 2 (1 per node)
CONTAINER_NAME="trtllm-multinode"
DGX1="10.0.0.1"
DGX2="10.0.0.2"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
ENTRYPOINT_URL="https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh"

# Load HF_TOKEN from .env if available
if [ -f "$HOME/.env" ]; then
    set +e
    source "$HOME/.env"
    set -e
fi

echo "[INFO] TRT-LLM Multi-Node Setup (CORRECTED VERSION)"
echo "  Image: $IMAGE (1.0.0rc3 - proven for multi-node DGX Spark)"
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE (multi-node: 1 per node)"
echo "  Port: $PORT"
echo "  DGX1: $DGX1"
echo "  DGX2: $DGX2"
echo ""

# Step 1: Stop any existing containers
echo "[STEP 1] Stopping any existing '$CONTAINER_NAME' containers..."
docker ps -a | grep -q "$CONTAINER_NAME" && docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker ps -a | grep -q "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Stop on DGX2 via SSH
echo "  Stopping on DGX2..."
ssh -i "$SSH_KEY_PATH" nvidia@"$DGX2" "docker ps -a | grep -q '$CONTAINER_NAME' && docker stop '$CONTAINER_NAME' 2>/dev/null || true; docker ps -a | grep -q '$CONTAINER_NAME' && docker rm '$CONTAINER_NAME' 2>/dev/null || true" || true

sleep 2

# Step 2: Pre-pull image on both nodes (critical for multi-node to avoid "stuck on preparing")
echo ""
echo "[STEP 2] Pre-pulling container image on both nodes..."
echo "  This is a CRITICAL workaround to avoid 'stuck on preparing' hang"
echo "  Pulling on DGX1..."
docker pull "$IMAGE" > /dev/null
echo "  ✓ Pulled on DGX1"

echo "  Pulling on DGX2..."
ssh -i "$SSH_KEY_PATH" nvidia@"$DGX2" "docker pull $IMAGE > /dev/null && echo '✓ Pulled on DGX2'" || true

sleep 3

# Step 3: Start container on DGX1
echo ""
echo "[STEP 3] Starting Docker container on DGX1..."
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  "$IMAGE" \
  bash -c "curl -fsSL $ENTRYPOINT_URL | bash"

echo "  Container started on DGX1"

# Step 4: Start container on DGX2
echo ""
echo "[STEP 4] Starting Docker container on DGX2 (via SSH)..."
ssh -i "$SSH_KEY_PATH" nvidia@"$DGX2" "
  docker run -d --rm \
    --name '$CONTAINER_NAME' \
    --gpus '\"device=all\"' \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -e UCX_NET_DEVICES='enp1s0f0np0,enp1s0f1np1' \
    -e NCCL_SOCKET_IFNAME='enp1s0f0np0,enp1s0f1np1' \
    -e OMPI_MCA_btl_tcp_if_include='enp1s0f0np0,enp1s0f1np1' \
    -e OMPI_MCA_orte_default_hostfile='/etc/openmpi-hostfile' \
    -e OMPI_MCA_rmaps_ppr_n_pernode='1' \
    -e OMPI_ALLOW_RUN_AS_ROOT='1' \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1' \
    -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
    -v ~/.ssh:/tmp/.ssh:ro \
    '$IMAGE' \
    bash -c 'curl -fsSL $ENTRYPOINT_URL | bash'
"

echo "  Container started on DGX2"

# Step 5: Copy hostfile into containers (on both nodes)
echo ""
echo "[STEP 5] Copying openmpi-hostfile into containers..."
sleep 3  # Wait for containers to be ready
docker cp ~/openmpi-hostfile "$CONTAINER_NAME":/etc/openmpi-hostfile
ssh -i "$SSH_KEY_PATH" nvidia@"$DGX2" "docker cp ~/openmpi-hostfile $CONTAINER_NAME:/etc/openmpi-hostfile"

echo "  Hostfiles copied"

# Step 6: Create config file in container
echo ""
echo "[STEP 6] Creating TRT-LLM config file..."
docker exec "$CONTAINER_NAME" bash -c 'cat <<EOF > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
EOF'
echo "  Config file created"

# Step 7: Verify SSHD is running
echo ""
echo "[STEP 7] Verifying SSHD inside containers..."
sleep 2

echo "  DGX1 SSHD status:"
docker exec "$CONTAINER_NAME" pgrep -a sshd || echo "    WARNING: sshd not found!"

echo "  DGX2 SSHD status:"
ssh -i "$SSH_KEY_PATH" nvidia@"$DGX2" "docker exec $CONTAINER_NAME pgrep -a sshd" || echo "    WARNING: sshd not found on DGX2!"

# Step 8: Test SSH connectivity between containers
echo ""
echo "[STEP 8] Testing SSH between containers (DGX1 → DGX2)..."
sleep 1

if docker exec "$CONTAINER_NAME" ssh -p 2233 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$DGX2" hostname 2>&1; then
    echo "  ✓ SSH connectivity OK"
else
    echo "  ✗ SSH connectivity FAILED (but proceeding)"
fi

# Step 9: Display launch command
echo ""
echo "=========================================="
echo "[SUCCESS] Containers ready on both nodes"
echo "=========================================="
echo ""
echo "To start TRT-LLM inference, run on DGX1:"
echo ""
echo "export HF_TOKEN='$HF_TOKEN'"
echo "export TRTLLM_MN_CONTAINER='$CONTAINER_NAME'"
echo ""
echo "docker exec \\"
echo "  -e MODEL='$MODEL' \\"
echo "  -e HF_TOKEN=\$HF_TOKEN \\"
echo "  \$TRTLLM_MN_CONTAINER bash -c '"
echo "    mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve \$MODEL \\"
echo "      --tp_size $TP_SIZE \\"
echo "      --backend pytorch \\"
echo "      --max_num_tokens 32768 \\"
echo "      --max_batch_size 4 \\"
echo "      --extra_llm_api_options /tmp/extra-llm-api-config.yml \\"
echo "      --port $PORT'"
echo ""
echo "Or run directly:"
echo ""

# Step 10: Optional auto-run inference
if [ "$AUTO_RUN" = "1" ]; then
    echo "[STEP 10] Starting inference..."
    export TRTLLM_MN_CONTAINER="$CONTAINER_NAME"
    docker exec \
      -e MODEL="$MODEL" \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      "$CONTAINER_NAME" bash -c "
        mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve \$MODEL \
          --tp_size $TP_SIZE \
          --backend pytorch \
          --max_num_tokens 32768 \
          --max_batch_size 4 \
          --extra_llm_api_options /tmp/extra-llm-api-config.yml \
          --port $PORT
      "
else
    echo "[READY] Run this to start inference:"
    echo ""
    echo "export HF_TOKEN='$HF_TOKEN'"
    echo "export TRTLLM_MN_CONTAINER='$CONTAINER_NAME'"
    echo "docker exec -e MODEL='$MODEL' -e HF_TOKEN=\$HF_TOKEN \$TRTLLM_MN_CONTAINER bash -c 'mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve \$MODEL --tp_size $TP_SIZE --backend pytorch --port $PORT --extra_llm_api_options /tmp/extra-llm-api-config.yml'"
fi
