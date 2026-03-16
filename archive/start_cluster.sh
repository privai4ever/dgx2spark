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
GPU_MEMORY_UTIL=0.75
PORT=8000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# Use NVIDIA's official vLLM 0.18.0 ARM64 image (certified for DGX)
VLLM_IMAGE="nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64"

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
# Auto-detect QSFP network interface
detect_network_interface() {
    # Try to auto-detect the QSFP interface (usually enp1s0f0np0 on DGX systems)
    # Fall back to the first interface with the 10.0.0.x network
    local iface

    # Check for known DGX QSFP interface names
    iface=$(ip link show | grep -E 'enp1s0f0np0|enp2s0f0np0|ens1f0np0' | grep -v ' DOWN' | head -1 | awk -F: '{print $2}' | xargs)

    if [ -z "$iface" ]; then
        # Fall back to interface that has the 10.0.0.x address
        iface=$(ip route get 10.0.0.2 2>/dev/null | head -1 | awk '{print $5}' | xargs)
    fi

    if [ -z "$iface" ]; then
        # Last resort: first non-lo interface with a 10.x.x.x address
        iface=$(ip -4 addr show | grep 'inet 10\.' | head -1 | awk '{print $NF}' | xargs)
    fi

    echo "${iface:-auto}"
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check if we can resolve the DGX2 IP
    if ! ping -c 1 -W 2 "$DGX2_IP" &>/dev/null; then
        log_error "Cannot reach DGX2 ($DGX2_IP). Check network connectivity."
        return 1
    fi

    # Check SSH connectivity to DGX2
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$DGX2_IP" "echo ok" 2>/dev/null | grep -q ok; then
        log_error "Cannot SSH to DGX2 ($DGX2_IP)."
        log_error "Make sure SSH keys are set up: ssh-copy-id $DGX2_IP"
        return 1
    fi

    # Check if Docker is running on both nodes
    if ! docker info &>/dev/null; then
        log_error "Docker is not running on DGX1. Start it with: sudo systemctl start docker"
        return 1
    fi

    if ! ssh "$DGX2_IP" "docker info" &>/dev/null; then
        log_error "Docker is not running on DGX2."
        return 1
    fi

    # Check GPU availability
    local local_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    local remote_gpus=$(ssh "$DGX2_IP" "nvidia-smi --list-gpus 2>/dev/null" | wc -l)

    if [ "$local_gpus" -lt 8 ]; then
        log_warn "DGX1 only has $local_gpus GPUs (expected 8)"
    fi

    if [ "$remote_gpus" -lt 8 ]; then
        log_warn "DGX2 only has $remote_gpus GPUs (expected 8)"
    fi

    log_info "Pre-flight checks passed (DGX1: ${local_gpus} GPUs, DGX2: ${remote_gpus} GPUs)"
    return 0
}

start_cluster() {
    log_info "Starting vLLM multi-node cluster..."
    log_info "Model: $MODEL"
    log_info "Tensor Parallel Size per node: $TENSOR_PARALLEL_SIZE"
    log_info "Total GPUs: $((TENSOR_PARALLEL_SIZE * 2))"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    # Auto-detect network interface
    # Prefer IB for 200G, fallback to QSFP Ethernet at 10G
    NCCL_IFACE=$(detect_network_interface)
    if ls /dev/infiniband 2>/dev/null | grep -q uverbs && grep -q "ib" <<< "$(ibv_devinfo 2>/dev/null | head -5 || echo "")"; then
        log_info "InfiniBand detected - using RDMA for 200Gbps!"
        NCCL_IFACE="ib0"
        export NCCL_IB_DISABLE=0
        export NCCL_IB_HCA=mlx5_0
        export NCCL_IB_GID_INDEX=3
        export NCCL_IB_TC=106
        export NCCL_ALGO=Ring
        export NCCL_MIN_NCHANNELS=1
        export NCCL_PROTO=LL
    else
        log_warn "InfiniBand not available - falling back to 10Gbps Ethernet over QSFP"
        export NCCL_IB_DISABLE=1
        export NCCL_ALGO=Tree
    fi
    log_info "Using network interface: ${NCCL_IFACE}"
    echo ""

    # Run pre-flight checks
    if ! preflight_checks; then
        log_error "Pre-flight checks failed. Aborting."
        return 1
    fi
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
        -e NCCL_SOCKET_IFNAME=${NCCL_IFACE:-auto} \
        -e GLOO_SOCKET_IFNAME=${NCCL_IFACE:-auto} \
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
        -e NCCL_SOCKET_IFNAME=${NCCL_IFACE:-auto} \
        -e GLOO_SOCKET_IFNAME=${NCCL_IFACE:-auto} \
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
