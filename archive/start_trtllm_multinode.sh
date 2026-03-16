#!/bin/bash
#
# NVIDIA TensorRT-LLM Multi-Node Startup Script for DGX Spark
# Uses trtllm-serve with distributed environment variables
# Target: Load large models across both DGX nodes
#

set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT=8000
MASTER_PORT=29500
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# Tensor parallelism: total GPUs across cluster
# Note: Must divide model's num_heads AND num_kv_heads. For Qwen2.5-7B (8 KV heads), max TP=8.
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}  # Default: 8 (4 GPUs per node * 2 nodes)

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
    echo "  --port PORT            API port (default: 30000)"
    echo "  --master-port PORT     Master port for distributed (default: 29500)"
    echo "  --stop                 Stop existing containers"
    echo "  --status               Show container status"
    echo "  --test-single          Test single-node only (WORLD_SIZE=1)"
    echo "  -h, --help             Show this help"
}

STOP=false
STATUS=false
TEST_SINGLE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --stop) STOP=true; shift ;;
        --status) STATUS=true; shift ;;
        --test-single) TEST_SINGLE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

stop_containers() {
    log_info "Stopping TRT-LLM containers..."
    docker stop trtllm-node-0 trtllm-node-1 2>/dev/null || true
    docker rm trtllm-node-0 trtllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop trtllm-node-0 trtllm-node-1 2>/dev/null || true; docker rm trtllm-node-0 trtllm-node-1 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=trtllm-node" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=trtllm-node' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers or SSH failed"
}

start_node() {
    local NODE_RANK=$1
    local IP=$2
    local CONTAINER_NAME="trtllm-node-${NODE_RANK}"

    # Determine if this is local or remote
    if [ "$IP" = "localhost" ] || [ "$IP" = "$(hostname -I | grep -o '10\.0\.0\.[0-9]*' | head -1 || echo '')" ]; then
        LOCAL=true
        DOCKER_CMD="docker"
    else
        LOCAL=false
        DOCKER_CMD="ssh $IP docker"
    fi

    # Build environment
    local ENV_VARS=(
        "-e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}"
        "-e HF_HOME=/root/.cache/huggingface"
        "-e NCCL_DEBUG=INFO"
        "-e NCCL_IB_HCA=mlx5_0"              # Mellanox HCA
        "-e NCCL_SOCKET_IFNAME=enp1s0f0np0"  # RoCE interface (NOT ib0)
        "-e NCCL_IB_DISABLE=0"               # Enable RDMA
        "-e NCCL_IB_GID_INDEX=3"             # GID index for RoCE v2
        "-e NCCL_IB_TC=106"                  # Traffic class for IB
        "-e NCCL_NET_GDR_LEVEL=2"
        "-e NCCL_P2P_DISABLE=1"
        "-e NCCL_ALGO=Ring"                  # Ring algorithm for IB
        "-e NCCL_MIN_NCHANNELS=1"            # Reduce channels for large messages
        "-e NCCL_PROTO=LL"                   # Low-latency protocol
    )
    # Add distributed env if multi-node
    if [ "$TEST_SINGLE" = false ]; then
        ENV_VARS+=(
            "-e WORLD_SIZE=2"
            "-e RANK=${NODE_RANK}"
            "-e MASTER_ADDR=${DGX1_IP}"
            "-e MASTER_PORT=${MASTER_PORT}"
        )
    fi

    # Build docker run command
    # Note: Not using --rm to keep container for debugging if it crashes
    local CMD="docker run -d --name ${CONTAINER_NAME} --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864"
    # Add IB device if available (for 200G RDMA)
    if ls /dev/infiniband 2>/dev/null | grep -q uverbs; then
        CMD="$CMD --device /dev/infiniband:/dev/infiniband"
    fi
    for var in "${ENV_VARS[@]}"; do
        CMD="$CMD $var"
    done
    CMD="$CMD -v ${HF_CACHE}:/root/.cache/huggingface"
    CMD="$CMD nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"
    CMD="$CMD trtllm-serve serve"
    CMD="$CMD ${MODEL}"
    CMD="$CMD --host 0.0.0.0"
    CMD="$CMD --port ${PORT}"
    CMD="$CMD --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    CMD="$CMD --max_seq_len 32768"
    CMD="$CMD --trust_remote_code"
    CMD="$CMD --log_level info"

    log_info "Starting TRT-LLM on ${IP} (node rank ${NODE_RANK})..."
    if [ "$LOCAL" = true ]; then
        eval "$CMD"
    else
        ssh "$IP" "$CMD"
    fi
}

start_cluster() {
    log_info "Starting TensorRT-LLM multi-node cluster..."
    log_info "Model: $MODEL"
    log_info "Master: $DGX1_IP (port $MASTER_PORT)"
    log_info "Slave: $DGX2_IP"
    log_info "API endpoint: http://${DGX1_IP}:${PORT}/v1/chat/completions"
    log_info "GPUs per node: 8 (total TP: ${TENSOR_PARALLEL_SIZE})"
    echo ""

    # Cleanup
    log_info "Cleaning up existing containers..."
    docker rm -f trtllm-node-0 trtllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker rm -f trtllm-node-0 trtllm-node-1 2>/dev/null || true" &
    wait
    sleep 2
    echo ""

    # Check if SSH to DGX2 works
    if ! ssh -o ConnectTimeout=5 "$DGX2_IP" "echo ok" 2>/dev/null | grep -q ok; then
        log_warn "Cannot SSH to DGX2 ($DGX2_IP). Multi-node will fail."
        log_warn "Test single-node only with --test-single"
        # fall through? maybe abort
    fi

    # Start node 0 on DGX1
    start_node 0 "$DGX1_IP"

    # Start node 1 on DGX2 (unless test-single)
    if [ "$TEST_SINGLE" = false ]; then
        log_info "Starting node 1 on remote DGX2..."
        start_node 1 "$DGX2_IP"
    else
        log_warn "Test-single mode: only starting node 0 on DGX1"
    fi

    echo ""
    log_info "Cluster starting..."
    log_info "API endpoint: http://${DGX1_IP}:${PORT}/v1/chat/completions"
    log_info "Check status: $0 --status"
    log_info "Stop cluster: $0 --stop"
}

# Main
if [ "$STATUS" = true ]; then
    show_status
elif [ "$STOP" = true ]; then
    stop_containers
else
    start_cluster
fi
