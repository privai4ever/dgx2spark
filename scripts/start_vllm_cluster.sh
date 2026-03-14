#!/bin/bash
#
# vLLM Distributed Cluster Startup Script
# Starts vLLM on multiple DGX Spark nodes for tensor-parallel inference
#

set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"

# Detect if running on DGX1
CURRENT_IP=$(hostname -I 2>/dev/null | grep -o '10\.0\.0\.[0-9]*' | head -1 || echo "")
RUNNING_ON_DGX1="${RUNNING_ON_DGX1:-false}"
if [ "$CURRENT_IP" = "10.0.0.1" ]; then
    RUNNING_ON_DGX1=true
fi
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.9
PORT=30000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"  # Set via env var
HF_CACHE="/home/ss/.cache/huggingface"
USE_SUDO="${USE_SUDO:-false}"

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
    echo "  --model MODEL          Model to load (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    echo "  --tp-size SIZE        Tensor parallel size (default: 2)"
    echo "  --port PORT           API port (default: 30000)"
    echo "  --stop                Stop existing containers"
    echo "  --status              Show container status"
    echo "  --sudo                Use sudo for docker commands"
    echo "  -h, --help            Show this help"
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
        --sudo) USE_SUDO=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

get_docker_cmd() {
    if [ "$USE_SUDO" = true ]; then
        echo "sudo docker"
    else
        echo "docker"
    fi
}

build_docker_cmd() {
    local rank=$1
    local ip=$2
    local DOCKER=$(get_docker_cmd)

    echo "${DOCKER} run -d --name vllm-node-${rank} \
  --gpus all --network host --ipc host \
  -v ${HF_CACHE}:/root/.cache/huggingface \
  -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  -e HF_HOME=/root/.cache/huggingface \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=2 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  vllm/vllm-openai:v0.6.3.post1 \
  --model ${MODEL} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --rank ${rank} \
  --master-address ${DGX1_IP} \
  --max-model-len ${MAX_MODEL_LEN} \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
  --port ${PORT}"
}

stop_containers() {
    log_info "Stopping vLLM containers..."
    local DOCKER=$(get_docker_cmd)

    $DOCKER stop vllm-node-0 vllm-node-1 2>/dev/null || true
    $DOCKER rm vllm-node-0 vllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "$DOCKER stop vllm-node-0 vllm-node-1 2>/dev/null || true; $DOCKER rm vllm-node-0 vllm-node-1 2>/dev/null || true" &

    wait
    log_info "Containers stopped"
}

show_status() {
    local DOCKER=$(get_docker_cmd)

    echo "=== DGX 1 ($DGX1_IP) ==="
    $DOCKER ps --filter "name=vllm-node" --format "table {{\.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "$DOCKER ps --filter 'name=vllm-node' --format 'table {{\.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

start_cluster() {
    log_info "Starting vLLM distributed cluster..."
    log_info "Model: $MODEL"
    log_info "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    log_info "Cleaning up existing containers..."
    docker rm -f vllm-node-0 vllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker rm -f vllm-node-0 vllm-node-1 2>/dev/null || true" &
    wait
    echo ""

    CMD_DGX1=$(build_docker_cmd 0 "$DGX1_IP")
    CMD_DGX2=$(build_docker_cmd 1 "$DGX2_IP")

    log_info "Starting vLLM on DGX 1 (master, rank 0)..."
    eval "$CMD_DGX1"

    log_info "Starting vLLM on DGX 2 (slave, rank 1)..."
    ssh "$DGX2_IP" "$CMD_DGX2"

    echo ""
    log_info "Cluster started!"
    log_info "API available at: http://${DGX1_IP}:${PORT}/v1/chat/completions"
}

if [ "$STATUS" = true ]; then
    show_status
elif [ "$STOP" = true ]; then
    stop_containers
else
    start_cluster
fi
