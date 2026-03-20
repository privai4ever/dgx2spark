#!/bin/bash

# vLLM Multi-Node Benchmark Setup (via Ray)
# Model: Nemotron-3-Super-120B-A12B-NVFP4
# Nodes: 2 DGX (10.0.0.1 and 10.0.0.2)
# Framework: vLLM with Ray distributed runtime
# Approach: Hybrid TP+PP to reduce inter-node communication

set -e

# Configuration
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
CONTAINER_NAME_1="vllm-nemotron-node1"
CONTAINER_NAME_2="vllm-nemotron-node2"
API_PORT="8002"
HF_TOKEN="${HF_TOKEN:-}"
TP_SIZE=4  # Tensor parallel per node
PP_SIZE=2  # Pipeline parallel across nodes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}vLLM Multi-Node Nemotron Setup (Hybrid TP+PP)${NC}"
echo "Model: $MODEL"
echo "TP Size: $TP_SIZE (per node)"
echo "PP Size: $PP_SIZE (across nodes)"
echo "DGX1: $DGX1_IP"
echo "DGX2: $DGX2_IP"
echo ""
echo -e "${YELLOW}WARNING: This is experimental. System-level GB10 bug may still cause crashes.${NC}"
echo ""

# Check connectivity
echo -e "${YELLOW}Checking connectivity to DGX2...${NC}"
if ! ping -c 1 $DGX2_IP > /dev/null 2>&1; then
    echo -e "${RED}Cannot reach DGX2 at $DGX2_IP${NC}"
    echo "Ensure network is configured (QSFP cables connected)"
    exit 1
fi
echo -e "${GREEN}✓ Connected to DGX2${NC}"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
fi

# Stop existing containers
echo -e "${YELLOW}Stopping old containers...${NC}"
docker stop $CONTAINER_NAME_1 2>/dev/null || true
docker rm $CONTAINER_NAME_1 2>/dev/null || true
ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker stop $CONTAINER_NAME_2 2>/dev/null || true; docker rm $CONTAINER_NAME_2 2>/dev/null || true" || true

# Start Ray Head on DGX1
echo -e "${YELLOW}Starting Ray head on DGX1...${NC}"
docker run \
    --name $CONTAINER_NAME_1 \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/vllm_logs:/tmp/vllm_logs \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e RAY_memory=100000000000 \
    -e RAY_object_store_memory=50000000000 \
    -p $API_PORT:8000 \
    --ipc=host \
    --shm-size 32g \
    -d \
    vllm/vllm-openai:latest \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --pipeline-parallel-size $PP_SIZE \
        --dtype float16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --port 8000 \
        --worker-use-ray

echo -e "${GREEN}Ray head started on DGX1${NC}"

# Start Ray worker on DGX2
echo -e "${YELLOW}Starting Ray worker on DGX2...${NC}"
ssh -o StrictHostKeyChecking=no root@$DGX2_IP << 'EOF'
docker run \
    --name vllm-nemotron-node2 \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/vllm_logs:/tmp/vllm_logs \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e RAY_memory=100000000000 \
    -e RAY_object_store_memory=50000000000 \
    -e RAY_HEAD_SERVICE_HOST=10.0.0.1 \
    -e RAY_HEAD_SERVICE_PORT=6379 \
    --ipc=host \
    --shm-size 32g \
    -d \
    vllm/vllm-openai:latest \
    ray start --address=10.0.0.1:6379 --resources='{"worker": 1}'
EOF

echo -e "${GREEN}Ray worker started on DGX2${NC}"

# Wait for API to be ready
echo -e "${YELLOW}Waiting for model to load (3-5 minutes)...${NC}"
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:$API_PORT/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo -e "${RED}✗ Timeout waiting for API${NC}"
    echo "Check logs:"
    echo "  DGX1: docker logs -f $CONTAINER_NAME_1"
    echo "  DGX2: ssh root@$DGX2_IP docker logs -f $CONTAINER_NAME_2"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "Multi-node vLLM running:"
echo "  DGX1 (head): $DGX1_IP:$API_PORT"
echo "  DGX2 (worker): $DGX2_IP (Ray worker)"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:$API_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}]}'"
echo ""
echo "Monitor Ray status:"
echo "  docker exec $CONTAINER_NAME_1 ray status"
echo ""
echo "View logs:"
echo "  DGX1: docker logs -f $CONTAINER_NAME_1"
echo "  DGX2: ssh root@$DGX2_IP docker logs -f $CONTAINER_NAME_2"
echo ""
echo "⚠️  Known Issue: GB10 multi-node has stability problems"
echo "    If it crashes, check NCCL logs for timeout errors"
