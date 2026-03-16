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

# NVIDIA Spark vLLM image (based on vLLM 0.12.0 with SM121 optimizations)
SPARK_VLLM_IMAGE="sparkarena/spark-vllm-docker:latest"

# NCCL env
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
    log_info "Stopping Spark vLLM containers..."
    docker stop spark-vllm-node-0 spark-vllm-node-1 2>/dev/null || true
    docker rm spark-vllm-node-0 spark-vllm-node-1 2>/dev/null || true
    ssh "$DGX2_IP" "docker stop spark-vllm-node-0 spark-vllm-node-1 2>/dev/null || true; docker rm spark-vllm-node-0 spark-vllm-node-1 2>/dev/null || true" &
    wait
    log_info "Containers stopped"
}

show_status() {
    echo "=== DGX 1 ($DGX1_IP) - LOCAL ==="
    docker ps --filter "name=spark-vllm-node" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "=== DGX 2 ($DGX2_IP) ==="
    ssh "$DGX2_IP" "docker ps --filter 'name=spark-vllm-node' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "No containers"
}

start_cluster() {
    log_info "Starting Spark vLLM distributed cluster..."
    log_info "Model: $MODEL"
    log_info "Master: $DGX1_IP"
    log_info "Slave: $DGX2_IP"
    echo ""

    # Cleanup
    log_info "Cleaning up existing containers..."
    for id in $(docker ps -aq --filter "name=spark-vllm-node" 2>/dev/null); do
        docker rm -f "$id" 2>/dev/null || true
    done
    ssh "$DGX2_IP" "for id in \$(docker ps -aq --filter 'name=spark-vllm-node' 2>/dev/null); do docker rm -f \$id 2>/dev/null || true; done" &
    wait
    sleep 2
    echo ""

    log_info "Starting Spark vLLM on DGX 1 (master, rank 0)..."
    docker run -d --name spark-vllm-node-0 \
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
        ${SPARK_VLLM_IMAGE} \
        python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL} \
            --tensor-parallel-size 8 \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
            --port ${PORT} \
            --trust-remote-code

    log_info "Starting Spark vLLM on DGX 2 (slave, rank 1)..."
    ssh "$DGX2_IP" "docker run -d --name spark-vllm-node-1 \
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
        ${SPARK_VLLM_IMAGE} \
        python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL} \
            --tensor-parallel-size 8 \
            --max-model-len ${MAX_MODEL_LEN} \
            --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
            --port ${PORT} \
            --trust-remote-code"

    echo ""
    log_info "Cluster started (2 independent nodes, use load balancer)!"
    log_info "API endpoints:"
    log_info "  DGX1: http://${DGX1_IP}:${PORT}/v1/chat/completions"
    log_info "  DGX2: http://${DGX2_IP}:${PORT}/v1/chat/completions"
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
