#!/bin/bash

# vLLM Cluster Benchmark Setup - Hybrid TP+PP
# Model: Nemotron-3-Super-120B-A12B-NVFP4
# Nodes: 2 DGX (10.0.0.1 and 10.0.0.2)
# Parallelism: TP=4 (within node) + PP=2 (across nodes)
# Framework: vLLM with Ray distributed runtime

set -e

# Configuration
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
CONTAINER_NAME_1="vllm-nemotron-head"
CONTAINER_NAME_2="vllm-nemotron-worker"
API_PORT="8002"
RAY_PORT="6379"
HF_TOKEN="${HF_TOKEN:-}"
TP_SIZE=4        # Tensor parallel per node (8 GPUs / 2 = 4)
PP_SIZE=2        # Pipeline parallel across nodes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  vLLM Multi-Node Cluster Benchmark         ║${NC}"
echo -e "${BLUE}║  Nemotron-3-Super-120B (NVFP4)             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:        $MODEL"
echo "  Nodes:        DGX1 ($DGX1_IP) + DGX2 ($DGX2_IP)"
echo "  TP Size:      $TP_SIZE (per node)"
echo "  PP Size:      $PP_SIZE (across nodes)"
echo "  API Port:     $API_PORT"
echo "  Ray Port:     $RAY_PORT"
echo ""
echo -e "${YELLOW}Why Hybrid TP+PP:${NC}"
echo "  • Pure TP=16:   All communication across network → crashes"
echo "  • TP=4/PP=2:    75% less inter-node traffic → more stable"
echo "  • PP=2:         Model split in half between nodes"
echo ""

# Check connectivity
echo -e "${YELLOW}[1/6] Checking cluster connectivity...${NC}"
if ! ping -c 1 $DGX2_IP > /dev/null 2>&1; then
    echo -e "${RED}✗ Cannot reach DGX2 at $DGX2_IP${NC}"
    echo "   Ensure QSFP cables connected and network configured"
    exit 1
fi
echo -e "${GREEN}✓ DGX1 ↔ DGX2 network OK${NC}"

# Check SSH
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$DGX2_IP "echo OK" > /dev/null 2>&1; then
    echo -e "${RED}✗ SSH to DGX2 failed${NC}"
    echo "   Run: ssh-keygen && ssh-copy-id root@$DGX2_IP"
    exit 1
fi
echo -e "${GREEN}✓ SSH to DGX2 OK${NC}"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠  HF_TOKEN not set (required for model download)${NC}"
    echo "   Set it: export HF_TOKEN=hf_..."
    exit 1
fi
echo -e "${GREEN}✓ HF_TOKEN configured${NC}"

# Stop existing containers
echo -e "${YELLOW}[2/6] Cleaning up old containers...${NC}"
docker stop $CONTAINER_NAME_1 2>/dev/null || true
docker rm $CONTAINER_NAME_1 2>/dev/null || true
ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker stop $CONTAINER_NAME_2 2>/dev/null || true; docker rm $CONTAINER_NAME_2 2>/dev/null || true" || true
sleep 2

# Pre-pull image on both nodes to save time
echo -e "${YELLOW}[3/6] Pre-pulling vLLM image on both nodes...${NC}"
docker pull vllm/vllm-openai:latest > /dev/null 2>&1 &
DGX1_PULL_PID=$!

ssh -o StrictHostKeyChecking=no root@$DGX2_IP "docker pull vllm/vllm-openai:latest > /dev/null 2>&1" > /dev/null 2>&1 &
DGX2_PULL_PID=$!

wait $DGX1_PULL_PID 2>/dev/null
wait $DGX2_PULL_PID 2>/dev/null
echo -e "${GREEN}✓ Images pulled${NC}"

# Start Ray head on DGX1
echo -e "${YELLOW}[4/6] Starting Ray head (DGX1)...${NC}"
docker run \
    --name $CONTAINER_NAME_1 \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e RAY_memory=100000000000 \
    -e RAY_object_store_memory=50000000000 \
    -p $API_PORT:8000 \
    -p $RAY_PORT:6379 \
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
        --max-num-seqs 256 \
        --port 8000 \
        --worker-use-ray \
        --ray-workers-use-nsight=false

echo -e "${GREEN}✓ Ray head started (waiting for initialization...)${NC}"
sleep 5

# Start Ray worker on DGX2
echo -e "${YELLOW}[5/6] Starting Ray worker (DGX2)...${NC}"
ssh -o StrictHostKeyChecking=no root@$DGX2_IP << REMOTE_CMD
docker run \
    --name $CONTAINER_NAME_2 \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e RAY_memory=100000000000 \
    -e RAY_object_store_memory=50000000000 \
    --ipc=host \
    --shm-size 32g \
    -d \
    vllm/vllm-openai:latest \
    python -c "
import ray
ray.init(
    address='$DGX1_IP:$RAY_PORT',
    runtime_env={'pip': []},
    resources={'worker': 1}
)
ray.shutdown()
"
REMOTE_CMD

echo -e "${GREEN}✓ Ray worker started${NC}"

# Wait for API to be ready
echo -e "${YELLOW}[6/6] Waiting for model to load and API to be ready...${NC}"
echo "      (Nemotron 120B: 5-10 minutes expected)"
echo ""

TIMEOUT=600
ELAPSED=0
LAST_STATUS=""

while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS=$(curl -s http://localhost:$API_PORT/v1/models 2>/dev/null | jq -r '.object // "error"' 2>/dev/null || echo "waiting")

    if [ "$STATUS" != "$LAST_STATUS" ]; then
        echo -n "."
        LAST_STATUS="$STATUS"
    fi

    if [ "$STATUS" = "list" ]; then
        echo ""
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))

    # Show progress every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo " ($ELAPSED/$TIMEOUT seconds)"
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo -e "${RED}✗ Timeout waiting for API${NC}"
    echo ""
    echo "Debugging:"
    echo "  DGX1 logs: docker logs -f $CONTAINER_NAME_1 | tail -50"
    echo "  DGX2 logs: ssh root@$DGX2_IP docker logs -f $CONTAINER_NAME_2 | tail -50"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Cluster Setup Complete!                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Cluster Status:${NC}"
echo "  API:          http://localhost:$API_PORT/v1"
echo "  Ray Head:     DGX1 ($DGX1_IP:$RAY_PORT)"
echo "  Ray Worker:   DGX2 ($DGX2_IP)"
echo ""
echo -e "${BLUE}Test Inference:${NC}"
echo "  curl -X POST http://localhost:$API_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"$MODEL\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": \"Explain quantum computing in 100 words\"}],"
echo "      \"max_tokens\": 256"
echo "    }'"
echo ""
echo -e "${BLUE}Monitoring:${NC}"
echo "  Ray status:       docker exec $CONTAINER_NAME_1 ray status"
echo "  Ray dashboard:    docker exec $CONTAINER_NAME_1 ray status --monitor"
echo "  DGX1 logs:        docker logs -f $CONTAINER_NAME_1"
echo "  DGX2 logs:        ssh root@$DGX2_IP docker logs -f $CONTAINER_NAME_2"
echo "  GPU memory:       docker stats --no-stream"
echo ""
echo -e "${BLUE}Load Testing:${NC}"
echo "  benchmarks/test_cluster_inference.sh"
echo ""
echo -e "${YELLOW}⚠  Known Issues:${NC}"
echo "  • GB10 multi-node may crash under extreme load"
echo "  • NCCL timeouts possible during prefill"
echo "  • Monitor logs for: 'NCCL', 'timeout', 'Out of memory'"
echo ""
echo -e "${BLUE}Cleanup:${NC}"
echo "  docker stop $CONTAINER_NAME_1"
echo "  ssh root@$DGX2_IP docker stop $CONTAINER_NAME_2"
