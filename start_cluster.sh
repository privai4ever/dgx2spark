#!/bin/bash
#
# vLLM Multi-Node Cluster Startup Script for DGX Spark
# Target: Load large models (200GB+) using tensor parallelism across both DGX nodes
# Uses custom vLLM build from GitHub with multi-node fixes
#

set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
# Tensor parallelism: 8 GPUs per node, 2 nodes = 16 GPUs total
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.9
PORT=30000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# Use custom vLLM image built from GitHub (has multi-node fix for 1+ GPU per node)
VLLM_IMAGE="dgx2spark/vllm:latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL          Model to load"
    echo "  --tp-size SIZE         Tensor parallel size per node (default: 8)"
    echo "  --port PORT            API port (default: 30000)"
    echo "  --stop                 Stop existing containers"
    echo "  --status               Show container status"
    echo "  -h, --help             Show this help"
}

STOP=false
STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --stop) STOP=true; shift ;;
        --status) STATUS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Stop containers
stop_containers() {
    log_info "Stopping vLLM containers..."
    docker stop dgx2spark-vllm-node-0 dgx2spark-vllm-node-1 2>/dev/null || true
    docker rm dgx2spark-vllm-node-0 dgx2spark-vllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop dgx2spark-vllm-node-0 dgx2spark-vllm-node-1 2>/dev/null || true; docker rm dgx2spark-vllm-node-0 dgx2spark-vllm-node-1 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

# Show status
show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=vllm-node" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=vllm-node' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

# Start cluster
start_cluster() {
    log_info "Starting vLLM multi-node cluster..."
    log_info "Model: $MODEL"
    log_info "Tensor Parallel Size per node: $TENSOR_PARALLEL_SIZE"
    log_info "Total GPUs: $((TENSOR_PARALLEL_SIZE * 2))"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    # Force remove existing containers
    log_info "Cleaning up existing containers..."
    for id in $(docker ps -aq --filter "name=vllm-node" 2>/dev/null); do
        docker rm -f "$id" 2>/dev/null || true
    done
    ssh "$DGX2_IP" "for id in \$(docker ps -aq --filter 'name=vllm-node' 2>/dev/null); do docker rm -f \$id 2>/dev/null || true; done" &
    wait
    sleep 2
    echo ""

    log_info "Starting vLLM on DGX 1 (node rank 0, TP=${TENSOR_PARALLEL_SIZE})..."
    docker run -d --name dgx2spark-vllm-node-0 \
        --gpus all --network host --ipc host \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=1 \
        -e NCCL_NET_GDR_LEVEL=2 \
        -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
        -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
        -e NCCL_P2P_DISABLE=1 \
        --env "VLLM_LOGGING_LEVEL=DEBUG" \
        ${VLLM_IMAGE} \
        vllm serve ${MODEL} \
            --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
            --nnodes 2 \
            --node-rank 0 \
            --master-addr ${DGX1_IP} \
            --master-port 29501 \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
            --port ${PORT}

    log_info "Starting vLLM on DGX 2 (node rank 1, TP=${TENSOR_PARALLEL_SIZE})..."
    ssh "$DGX2_IP" "docker run -d --name dgx2spark-vllm-node-1 \
        --gpus all --network host --ipc host \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=1 \
        -e NCCL_NET_GDR_LEVEL=2 \
        -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
        -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
        -e NCCL_P2P_DISABLE=1 \
        --env \"VLLM_LOGGING_LEVEL=DEBUG\" \
        ${VLLM_IMAGE} \
        vllm serve ${MODEL} \
            --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
            --nnodes 2 \
            --node-rank 1 \
            --master-addr ${DGX1_IP} \
            --master-port 29501 \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
            --port ${PORT}"

    echo ""
    log_info "Cluster started!"
    log_info "API available at: http://${DGX1_IP}:${PORT}/v1/chat/completions"
    log_info ""
    log_info "Check status with: $0 --status"
    log_info "Stop cluster with: $0 --stop"
}

# Main
if [ "$STATUS" = true ]; then
    show_status
elif [ "$STOP" = true ]; then
    stop_containers
else
    start_cluster
fi
