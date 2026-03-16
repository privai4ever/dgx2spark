#!/bin/bash
#
# TensorRT-LLM Multi-Node using NVIDIA's official method
# BUT with mpirun on HOST (not inside container) to avoid SSH issues
#
# Based on: https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm
#

set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
MODEL="${MODEL:-nvidia/Llama-3.1-8B-Instruct-FP4}"
PORT=8000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# TP size: total GPUs across cluster
# For 2 nodes × 8 GPUs = 16 total
TP_SIZE=${TP_SIZE:-16}

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
    echo "  --model MODEL          Model to load (default: nvidia/Llama-3.1-8B-Instruct-FP4)"
    echo "  --tp-size SIZE         Total tensor parallel size (default: 16)"
    echo "  --port PORT            API port (default: 8000)"
    echo "  --stop                 Stop existing containers"
    echo "  --status               Show container status"
    echo "  -h, --help             Show this help"
}

STOP=false
STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --stop) STOP=true; shift ;;
        --status) STATUS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

stop_containers() {
    log_info "Stopping TRT-LLM multi-node containers..."
    docker stop trtllm-multinode 2>/dev/null || true
    docker rm trtllm-multinode 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop trtllm-multinode 2>/dev/null || true; docker rm trtllm-multinode 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=trtllm-multinode" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=trtllm-multinode' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

start_node() {
    local NODE_IP=$1

    # Check if this is local or remote
    if [ "$NODE_IP" = "$DGX1_IP" ]; then
        DOCKER_CMD="docker"
    else
        DOCKER_CMD="ssh $NODE_IP docker"
    fi

    # Clean up old container
    $DOCKER_CMD ps -a --filter "name=trtllm-multinode" --format "{{.Names}}" | grep -q trtllm-multinode && \
        $DOCKER_CMD rm -f trtllm-multinode 2>/dev/null || true

    # Download entrypoint script to host (not container)
    log_info "[$NODE_IP] Preparing entrypoint script..."
    if [ "$NODE_IP" = "$DGX1_IP" ]; then
        curl -s https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh \
            > /tmp/trtllm-mn-entrypoint.sh 2>/dev/null || true
        chmod +x /tmp/trtllm-mn-entrypoint.sh
    else
        ssh $NODE_IP "curl -s https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh > /tmp/trtllm-mn-entrypoint.sh && chmod +x /tmp/trtllm-mn-entrypoint.sh" 2>/dev/null &
    fi
    wait

    # Start container with NVIDIA's official multi-node setup
    log_info "[$NODE_IP] Starting TRT-LLM multi-node container..."

    $DOCKER_CMD run -d --name trtllm-multinode \
        --gpus all \
        --network host \
        --ipc host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
        -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
        -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
        -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
        -e OMPI_ALLOW_RUN_AS_ROOT="1" \
        -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
        -v $HF_CACHE:/root/.cache/huggingface/ \
        -v ~/.ssh:/tmp/.ssh:ro \
        -v /tmp/trtllm-mn-entrypoint.sh:/usr/local/bin/trtllm-mn-entrypoint.sh:ro \
        -v /home/nvidia/code/dgx2spark/openmpi-hostfile:/etc/openmpi-hostfile:ro \
        nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
        /usr/local/bin/trtllm-mn-entrypoint.sh &
}

create_hostfile() {
    log_info "Creating OpenMPI hostfile..."
    local HOSTFILE_PATH="/home/nvidia/code/dgx2spark/openmpi-hostfile"
    cat > $HOSTFILE_PATH <<EOF
$DGX1_IP slots=8
$DGX2_IP slots=8
EOF
    log_info "Hostfile created:"
    cat $HOSTFILE_PATH
    export OMPI_MCA_orte_default_hostfile="$HOSTFILE_PATH"
}

download_model() {
    log_info "Downloading model: $MODEL"
    log_info "Using mpirun on HOST (not inside container)"

    mpirun \
        --hostfile /etc/openmpi-hostfile \
        --oversubscribe \
        -x HF_TOKEN \
        bash -c "docker exec trtllm-multinode hf download $MODEL" || {
        log_warn "Model download failed or already exists"
        return 1
    }
    log_info "Model download completed"
}

start_server() {
    log_info "Starting TRT-LLM server with mpirun on host..."
    log_info "Command: mpirun --hostfile /etc/openmpi-hostfile --oversubscribe -x HF_TOKEN \\"
    log_info "  docker exec trtllm-multinode \\"
    log_info "  trtllm-llmapi-launch trtllm-serve $MODEL \\"
    log_info "  --tp_size $TP_SIZE --backend pytorch \\"
    log_info "  --max_num_tokens 32768 --max_batch_size 4 \\"
    log_info "  --extra_llm_api_options /tmp/extra-llm-api-config.yml \\"
    log_info "  --port $PORT"
    echo ""

    # Create config file in container (on DGX1, which is primary)
    docker exec trtllm-multinode bash -c "cat <<'EOF' > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: 'auto'
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    "

    # Start server via mpirun on host
    mpirun \
        --hostfile /etc/openmpi-hostfile \
        --oversubscribe \
        -x HF_TOKEN \
        -np $TP_SIZE \
        bash -c "docker exec trtllm-multinode trtllm-llmapi-launch trtllm-serve \$MODEL \
          --tp_size $TP_SIZE \
          --backend pytorch \
          --max_num_tokens 32768 \
          --max_batch_size 4 \
          --extra_llm_api_options /tmp/extra-llm-api-config.yml \
          --port $PORT" &

    # Wait for server to start
    log_info "Waiting for server to start on port $PORT..."
    for i in {1..120}; do
        if ss -tlnp | grep -q ":$PORT "; then
            break
        fi
        sleep 2
    done

    if ! ss -tlnp | grep -q ":$PORT "; then
        log_error "Server failed to start on port $PORT"
        return 1
    fi

    echo ""
    log_info "=========================================="
    log_info "TensorRT-LLM Multi-Node Cluster Started!"
    log_info "=========================================="
    log_info "Model: $MODEL"
    log_info "TP Size: $TP_SIZE"
    log_info "API endpoint: http://$DGX1_IP:$PORT/v1/chat/completions"
    echo ""
    log_info "Check status: $0 --status"
    log_info "Stop cluster: $0 --stop"
    log_info ""
    log_info "View logs (primary node): docker logs -f trtllm-multinode"
}

start_cluster() {
    log_info "Starting TensorRT-LLM multi-node cluster (NVIDIA method + host mpirun)..."
    log_info "Model: $MODEL"
    log_info "TP Size: $TP_SIZE"
    log_info "Nodes: $DGX1_IP (primary), $DGX2_IP (worker)"
    log_info "API port: $PORT"
    echo ""

    # Pre-flight checks
    log_info "Running pre-flight checks..."

    if ! ssh -o ConnectTimeout=5 "$DGX2_IP" "echo ok" 2>/dev/null | grep -q ok; then
        log_error "Cannot SSH to DGX2 ($DGX2_IP). Set up SSH keys first!"
        return 1
    fi

    if ! docker info &>/dev/null; then
        log_error "Docker not running on DGX1"
        return 1
    fi

    if ! ssh "$DGX2_IP" "docker info" &>/dev/null; then
        log_error "Docker not running on DGX2"
        return 1
    fi

    # Create hostfile
    create_hostfile
    echo ""

    # Stop existing containers
    log_info "Cleaning up existing containers..."
    stop_containers
    sleep 2
    echo ""

    # Start containers on BOTH nodes
    log_info "Starting TRT-LLM containers on both nodes..."
    start_node "$DGX1_IP" &
    start_node "$DGX2_IP" &
    wait
    echo ""

    # Wait for containers to be ready
    log_info "Waiting for containers to start..."
    for i in {1..30}; do
        if docker ps | grep -q trtllm-multinode; then
            break
        fi
        sleep 2
    done

    if ! docker ps | grep -q trtllm-multinode; then
        log_error "Container on DGX1 failed to start"
        return 1
    fi

    if ! ssh "$DGX2_IP" "docker ps | grep -q trtllm-multinode"; then
        log_error "Container on DGX2 failed to start"
        return 1
    fi

    echo ""
    log_info "Containers are running on both nodes!"
    echo ""

    # Download model
    download_model
    echo ""

    # Start server
    start_server
}

# Main
if [ "$STATUS" = true ]; then
    show_status
elif [ "$STOP" = true ]; then
    stop_containers
else
    start_cluster
fi
