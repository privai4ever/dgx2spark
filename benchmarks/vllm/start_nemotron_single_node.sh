#!/bin/bash

# vLLM Single-Node Benchmark Setup
# Model: Nemotron-3-Super-120B-A12B-NVFP4
# GPUs: 8 (single DGX node)
# Framework: vLLM

set -e

# Configuration
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
CONTAINER_NAME="vllm-nemotron-single"
API_PORT="8001"
HF_TOKEN="${HF_TOKEN:-}"
TP_SIZE=8

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}vLLM Single-Node Nemotron Setup${NC}"
echo "Model: $MODEL"
echo "TP Size: $TP_SIZE"
echo "Port: $API_PORT"
echo "Container: $CONTAINER_NAME"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. Large model downloads may fail.${NC}"
    echo "Set it with: export HF_TOKEN=your_token"
    # Still continue - might be cached
fi

# Stop existing container
echo -e "${YELLOW}Stopping old container...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Start vLLM container
echo -e "${YELLOW}Starting vLLM container...${NC}"
docker run \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/vllm_logs:/tmp/vllm_logs \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_LOGGING_LEVEL=INFO \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -p $API_PORT:8000 \
    --ipc=host \
    --shm-size 32g \
    -d \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --tensor-parallel-size $TP_SIZE \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --max-model-len 4096 \
    --port 8000

echo -e "${GREEN}Container started!${NC}"
echo "Waiting for model to load (2-3 minutes)..."

# Wait for API to be ready
TIMEOUT=180
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
    echo "Check logs with: docker logs -f $CONTAINER_NAME"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "API endpoint: http://localhost:$API_PORT/v1"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:$API_PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}]}'"
echo ""
echo "View logs: docker logs -f $CONTAINER_NAME"
echo "Stop: docker stop $CONTAINER_NAME"
