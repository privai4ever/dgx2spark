#!/bin/bash

# TRT-LLM Cluster Benchmark Setup - Pure Tensor Parallelism
# Model: Nemotron-3-Super-120B-A12B-NVFP4
# Nodes: 2 DGX (10.0.0.1 and 10.0.0.2)
# Parallelism: TP=8 (across all 16 GPUs)
# Framework: TensorRT-LLM with mpirun distributed execution

set -e

# Configuration
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
CONTAINER_NAME="trtllm-nemotron-cluster"
API_PORT="8004"
HF_TOKEN="${HF_TOKEN:-}"
TP_SIZE=8        # Tensor parallel across all 16 GPUs (8 per node)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TRT-LLM Multi-Node Cluster Benchmark      ║${NC}"
echo -e "${BLUE}║  Nemotron-3-Super-120B (NVFP4)             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:        $MODEL"
echo "  Nodes:        DGX1 ($DGX1_IP) + DGX2 ($DGX2_IP)"
echo "  TP Size:      $TP_SIZE (16 GPUs total, 8 per node)"
echo "  API Port:     $API_PORT"
echo "  Framework:    TensorRT-LLM 1.0.0rc3"
echo ""
echo -e "${YELLOW}Approach:${NC}"
echo "  • Pure tensor parallelism across 16 GPUs"
echo "  • Heavy inter-node communication"
echo "  • Higher performance IF it doesn't crash"
echo "  • Known unstable on GB10 under load"
echo ""
echo -e "${RED}WARNING:${NC} This setup may crash due to GB10 NCCL bugs"
echo "If it fails, try: benchmarks/vllm/start_nemotron_cluster_tp4pp2.sh"
echo ""

# Check connectivity
echo -e "${YELLOW}[1/5] Checking cluster connectivity...${NC}"
if ! ping -c 1 $DGX2_IP > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot reach DGX2 at $DGX2_IP${NC}"
    exit 1
fi
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$DGX2_IP "echo OK" > /dev/null 2>&1; then
    echo -e "${RED}✗ SSH to DGX2 failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Cluster connectivity OK${NC}"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}✗ HF_TOKEN not set${NC}"
    exit 1
fi
echo -e "${GREEN}✓ HF_TOKEN configured${NC}"

# Stop existing containers
echo -e "${YELLOW}[2/5] Cleaning up old containers...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker stop $CONTAINER_NAME 2>/dev/null || true; docker rm $CONTAINER_NAME 2>/dev/null || true" || true
sleep 2

# Create OpenMPI hostfile
echo -e "${YELLOW}[3/5] Creating OpenMPI configuration...${NC}"
cat > /tmp/openmpi-hostfile << EOF
$DGX1_IP slots=1
$DGX2_IP slots=1
EOF
echo -e "${GREEN}✓ Hostfile created${NC}"

# Start TRT-LLM containers on both nodes
echo -e "${YELLOW}[4/5] Starting TRT-LLM containers on both nodes...${NC}"

# DGX1
docker run \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/openmpi-hostfile:/etc/openmpi-hostfile \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e UCX_NET_DEVICES=enp1s0f0np0,enp1s0f1np1 \
    -e OMPI_MCA_btl_tcp_if_include=enp1s0f0np0,enp1s0f1np1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -p $API_PORT:8000 \
    --ipc=host \
    --shm-size 32g \
    -d \
    nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
    sleep infinity

echo -e "${GREEN}✓ DGX1 container started${NC}"

# DGX2 (same container)
ssh -o StrictHostKeyChecking=no root@$DGX2_IP << REMOTE_CMD
docker run \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/openmpi-hostfile:/etc/openmpi-hostfile \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e UCX_NET_DEVICES=enp1s0f0np0,enp1s0f1np1 \
    -e OMPI_MCA_btl_tcp_if_include=enp1s0f0np0,enp1s0f1np1 \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    --ipc=host \
    --shm-size 32g \
    -d \
    nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
    sleep infinity
REMOTE_CMD

echo -e "${GREEN}✓ DGX2 container started${NC}"

# Copy hostfile to both containers
docker cp /tmp/openmpi-hostfile $CONTAINER_NAME:/etc/openmpi-hostfile
ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker cp /etc/openmpi-hostfile $CONTAINER_NAME:/etc/openmpi-hostfile" || true

sleep 3

# Launch inference
echo -e "${YELLOW}[5/5] Launching TRT-LLM distributed inference...${NC}"
echo "      This will download/compile model and start serving"
echo ""

docker exec \
    -e HF_TOKEN="$HF_TOKEN" \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
    -e UCX_NET_DEVICES=enp1s0f0np0,enp1s0f1np1 \
    $CONTAINER_NAME bash -c '
    mpirun \
        -x HF_TOKEN \
        -x NCCL_SOCKET_IFNAME=enp1s0f0np0,enp1s0f1np1 \
        -x UCX_NET_DEVICES=enp1s0f0np0,enp1s0f1np1 \
        -hostfile /etc/openmpi-hostfile \
        -np 2 \
        trtllm-llmapi-launch trtllm-serve '"'"'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4'"'"' \
            --tp_size 8 \
            --backend pytorch \
            --max_num_tokens 32768 \
            --max_batch_size 4 \
            --port 8000
' &

INFERENCE_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}Waiting for model to load (10-15 minutes)...${NC}"

TIMEOUT=900
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:$API_PORT/v1/models > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))

    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo " ($ELAPSED/$TIMEOUT seconds)"
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo -e "${RED}✗ Timeout waiting for API${NC}"
    echo ""
    echo "Check logs:"
    echo "  docker logs $CONTAINER_NAME | tail -100"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Cluster Setup Complete!                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Cluster Status:${NC}"
echo "  API:          http://localhost:$API_PORT/v1"
echo "  DGX1:         $DGX1_IP (API endpoint)"
echo "  DGX2:         $DGX2_IP (worker)"
echo "  Tensor Size:  $TP_SIZE GPUs"
echo ""
echo -e "${BLUE}Test Inference:${NC}"
echo "  curl -X POST http://localhost:$API_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"$MODEL\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": \"What is quantum computing?\"}],"
echo "      \"max_tokens\": 256"
echo "    }'"
echo ""
echo -e "${BLUE}Monitoring:${NC}"
echo "  Logs:         docker logs -f $CONTAINER_NAME"
echo "  NCCL debug:   docker logs $CONTAINER_NAME | grep -i nccl"
echo "  GPU memory:   docker stats --no-stream"
echo ""
echo -e "${YELLOW}⚠  If crashes:${NC}"
echo "  1. Check NCCL timeout errors in logs"
echo "  2. Try hybrid approach: benchmarks/vllm/start_nemotron_cluster_tp4pp2.sh"
echo "  3. This is a known GB10 issue, not TRT-LLM bug"
