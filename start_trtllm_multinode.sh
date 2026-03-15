#!/bin/bash
#
# NVIDIA TensorRT-LLM Multi-Node Startup Script for DGX Spark
# Based on NVIDIA's official dgx-spark-playbooks
# Target: Load large models (200GB+) across both DGX nodes
#

set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT=30000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# NVIDIA TRT-LLM image
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"

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
    echo "  --stop                 Stop existing containers"
    echo "  --status               Show container status"
    echo "  -h, --help             Show this help"
}

STOP=false
STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --stop) STOP=true; shift ;;
        --status) STATUS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Stop containers
stop_containers() {
    log_info "Stopping TRT-LLM containers..."
    docker stop trtllm-multinode 2>/dev/null || true
    docker rm trtllm-multinode 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop trtllm-multinode 2>/dev/null || true; docker rm trtllm-multinode 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

# Show status
show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=trtllm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=trtllm' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

# Start cluster
start_cluster() {
    log_info "Starting TensorRT-LLM multi-node cluster..."
    log_info "Model: $MODEL"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    # Cleanup
    log_info "Cleaning up existing containers..."
    docker rm -f trtllm-multinode 2>/dev/null || true
    ssh "$DGX2_IP" "docker rm -f trtllm-multinode 2>/dev/null || true" &
    wait
    sleep 2
    echo ""

    log_info "Starting TRT-LLM on DGX 1 (master)..."
    docker run -d --name trtllm-multinode \
        --rm \
        --gpus all \
        --network host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
        -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
        -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
        -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
        -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
        -e OMPI_ALLOW_RUN_AS_ROOT="1" \
        -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
        -e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_HCA="mlx5_bond" \
        ${TRTLLM_IMAGE} \
        sh -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | sh"

    log_info "Starting TRT-LLM on DGX 2 (slave)..."
    ssh "$DGX2_IP" "docker run -d --name trtllm-multinode \
        --rm \
        --gpus all \
        --network host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e UCX_NET_DEVICES=\"enp1s0f0np0,enp1s0f1np1\" \
        -e NCCL_SOCKET_IFNAME=\"enp1s0f0np0,enp1s0f1np1\" \
        -e OMPI_MCA_btl_tcp_if_include=\"enp1s0f0np0,enp1s0f1np1\" \
        -e OMPI_MCA_orte_default_hostfile=\"/etc/openmpi-hostfile\" \
        -e OMPI_MCA_rmaps_ppr_n_pernode=\"1\" \
        -e OMPI_ALLOW_RUN_AS_ROOT=\"1\" \
        -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=\"1\" \
        -e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_HCA=\"mlx5_bond\" \
        ${TRTLLM_IMAGE} \
        sh -c \"curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | sh\""

    echo ""
    log_info "Cluster starting..."
    log_info "Check status with: $0 --status"
    log_info "Stop cluster with: $0 --stop"
    log_info ""
    log_info "NOTE: The entrypoint script will handle model download and engine startup automatically."
}

# Main
if [ "$STATUS" = true ]; then
    show_status
elif [ "$STOP" = true ]; then
    stop_containers
else
    start_cluster
fi
