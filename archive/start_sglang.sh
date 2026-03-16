#!/bin/bash
set -e

# Configuration
DGX1_IP="10.0.0.1"
DGX2_IP="10.0.0.2"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.9
PORT=30000
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_CACHE="/home/ss/.cache/huggingface"

# sGLang image
SGLANG_IMAGE="sglang/sglang:latest"

# NCCL env (same as vLLM)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_P2P_DISABLE=1

log_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

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

stop_containers() {
    log_info "Stopping sGLang containers..."
    docker stop sglang-node-0 sglang-node-1 2>/dev/null || true
    docker rm sglang-node-0 sglang-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop sglang-node-0 sglang-node-1 2>/dev/null || true; docker rm sglang-node-0 sglang-node-1 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=sglang-node" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=sglang-node' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

start_cluster() {
    log_info "Starting sGLang distributed cluster..."
    log_info "Model: $MODEL"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    # Cleanup
    log_info "Cleaning up existing containers..."
    for id in $(docker ps -aq --filter "name=sglang-node" 2>/dev/null); do
        docker rm -f "$id" 2>/dev/null || true
    done
    ssh "$DGX2_IP" "for id in \$(docker ps -aq --filter 'name=sglang-node' 2>/dev/null); do docker rm -f \$id 2>/dev/null || true; done" &
    wait
    sleep 2
    echo ""

    log_info "Starting sGLang on DGX 1 (master, rank 0)..."
    docker run -d --name sglang-node-0 \
        --gpus all --network host --ipc host \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
        -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
        -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \
        -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        ${SGLANG_IMAGE} \
        python3 -m sglang.launch --model ${MODEL} --tp-size 1 --nnodes 2 --node-rank 0 --master-addr ${DGX1_IP} --master-port 29500 --port ${PORT} --max-seq-len ${MAX_MODEL_LEN} --trust-remote-code

    log_info "Starting sGLang on DGX 2 (slave, rank 1)..."
    ssh "$DGX2_IP" "docker run -d --name sglang-node-1 \
        --gpus all --network host --ipc host \
        -v ${HF_CACHE}:/root/.cache/huggingface \
        -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
        -e HF_HOME=/root/.cache/huggingface \
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
        -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
        -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \
        -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        ${SGLANG_IMAGE} \
        python3 -m sglang.launch --model ${MODEL} --tp-size 1 --nnodes 2 --node-rank 1 --master-addr ${DGX1_IP} --master-port 29500 --port ${PORT} --max-seq-len ${MAX_MODEL_LEN} --trust-remote-code"

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
