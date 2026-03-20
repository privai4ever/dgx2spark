# DGX Spark Cluster Inference Benchmark Suite

This directory contains startup scripts for benchmarking large language model inference on a **two-node DGX Spark cluster** (10.0.0.1 and 10.0.0.2).

**Focus:** Multi-node distributed inference only (single-node is not interesting)

## Cluster Setups

| Framework | Approach | Status | Purpose |
|-----------|----------|--------|---------|
| **vLLM Ray** | Hybrid TP=4/PP=2 | ✅ Recommended | Stable, reduced inter-node traffic |
| **TRT-LLM mpirun** | Pure TP=8 | ⚠️ Risky | Maximum performance (may crash) |
| **SGLang** | (Coming soon) | 🔄 WIP | Alternative to vLLM |

---

## Quick Start

### Recommended: vLLM with Hybrid TP+PP

```bash
cd benchmarks/vllm
chmod +x start_nemotron_cluster_tp4pp2.sh
./start_nemotron_cluster_tp4pp2.sh
```

**Why:** Reduces inter-node communication by 75%, more stable on GB10.

### Aggressive: TRT-LLM with Pure Tensor Parallelism

```bash
cd benchmarks/trtllm
chmod +x start_nemotron_cluster_tp8.sh
./start_nemotron_cluster_tp8.sh
```

**Why:** Maximum performance, but known to crash on GB10 under load.

---

## Setup Details

### vLLM Single-Node

**File:** `vllm/start_nemotron_single_node.sh`

**What it does:**
- Starts vLLM container with 8 GPUs (full DGX node)
- Loads Nemotron NVFP4 (60GB, fits with TP=8)
- Exposes OpenAI-compatible API on port 8001

**Configuration:**
```bash
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
TP_SIZE=8
GPU_MEMORY_UTIL=0.90
```

**API Test:**
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64
  }'
```

---

### vLLM Multi-Node (Ray)

**File:** `vllm/start_nemotron_multinode.sh`

**What it does:**
- Starts Ray head on DGX1 (10.0.0.1)
- Starts Ray worker on DGX2 (10.0.0.2)
- Uses Hybrid Tensor Parallel + Pipeline Parallel (TP=4, PP=2)
- Reduces inter-node communication by splitting model differently

**Why Hybrid TP+PP:**
- Pure TP=16: All communication is across network → GB10 crashes
- Hybrid TP=4/PP=2: 75% less inter-node traffic → more stable

**Configuration:**
```bash
TP_SIZE=4        # Tensor parallel per node
PP_SIZE=2        # Pipeline parallel across nodes
GPU_MEMORY_UTIL=0.85
```

**Expected Behavior:**
- DGX1 receives API requests
- Ray distributes computation across both nodes
- Model split with PP (layers on different nodes)
- Within-node communication via NVLink (fast)
- Cross-node communication via RoCE (slow, hence fewer syncs)

**Monitoring:**
```bash
# Check Ray cluster status
docker exec vllm-nemotron-node1 ray status

# Watch logs on DGX1
docker logs -f vllm-nemotron-node1

# Watch logs on DGX2
ssh root@10.0.0.2 docker logs -f vllm-nemotron-node2
```

---

### SGLang Single-Node

**File:** `sglang/start_nemotron_single_node.sh`

**What it does:**
- Similar to vLLM but uses SGLang runtime
- Optimized for low-latency inference
- Also exposes OpenAI-compatible API on port 8003

**Configuration:**
```bash
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
TP_SIZE=8
GPU_MEMORY_UTIL=0.90
```

**When to use SGLang:**
- Better latency for some workloads
- Experimental multi-node support via Ray (not included yet)
- Good for comparing against vLLM

---

### TRT-LLM (Legacy)

**File:** `../start_trtllm_correct.sh`

**Status:** ⚠️ Multi-node unstable on GB10 due to NCCL timeout issues

See `../FINAL_SOLUTION.md` and `../NEMOTRON_MULTINODE_RESEARCH.md` for details.

---

## Benchmark Workflow

### Phase 1: Single-Node Baseline
```bash
# Test vLLM
./vllm/start_nemotron_single_node.sh
# Record throughput, latency, memory usage
docker stats vllm-nemotron-single
```

### Phase 2: Stress Test
```bash
# Generate load with concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8001/v1/chat/completions ... &
done
wait
```

### Phase 3: Multi-Node (if Phase 1 passes)
```bash
# Only attempt if single-node is stable
./vllm/start_nemotron_multinode.sh
# Monitor for NCCL timeouts
docker logs vllm-nemotron-node1 | grep -i "nccl\|timeout\|error"
```

---

## Model Variants

The scripts default to **NVFP4** (60GB) because:
- ✅ Fits on single node (8×80GB = 640GB)
- ✅ Fits on two nodes with TP/PP
- ✅ Good quality/speed tradeoff

**Other variants available:**

| Variant | Size | TP=8 Single-Node | TP=4/PP=2 Multi-Node | Notes |
|---------|------|--|--|--|
| **NVFP4** | 60GB | ✅ Yes | ✅ Yes | Best for GB10 |
| **FP8** | 120GB | ❓ Tight | ✅ Yes | Needs `gpu_memory_util=0.75` |
| **BF16** | 240GB | ❌ No | ❌ No | Requires H100 cluster |

To use different variant, edit the script and change:
```bash
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
```

---

## Troubleshooting

### Container won't start
```bash
# Check Docker daemon
docker ps

# Check GPU availability
docker run --rm --runtime nvidia --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi
```

### Model loading hangs
```bash
# Increase timeout in script (default 180s)
# Check if HuggingFace download is stuck
docker logs -f container-name | grep -i "download\|loading"
```

### API not responding
```bash
# Check if container is still running
docker ps | grep nemotron

# View full logs
docker logs container-name

# Common issues:
# - Out of memory: reduce gpu_memory_utilization
# - Model not found: check HF_TOKEN
# - Port conflict: change API_PORT in script
```

### Multi-node connectivity issues
```bash
# Test SSH between nodes
ssh -o StrictHostKeyChecking=no root@10.0.0.2 "nvidia-smi"

# Check network
ping 10.0.0.2
iperf3 -c 10.0.0.2

# Verify QSFP cables connected
ethtool -i enp1s0f0np0 | grep driver
```

### NCCL timeouts (multi-node)
```bash
# This is the GB10 multi-node bug
# Options:
# 1. Reduce compute load (lower batch size)
# 2. Use single-node instead
# 3. Contact NVIDIA support with logs
docker logs vllm-nemotron-node1 | grep -i "nccl"
```

---

## Performance Monitoring

### Real-time monitoring
```bash
docker stats --no-stream \
  vllm-nemotron-single \
  sglang-nemotron-single

# Watch specific metrics:
# CONTAINER ID        NAME                    CPU %    MEM USAGE / LIMIT
# Shows GPU memory usage and utilization
```

### Throughput testing
```bash
# Simple load test (10 requests)
time for i in {1..10}; do
  curl -X POST http://localhost:8001/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
      "messages": [{"role": "user", "content": "Hello, how are you?"}],
      "max_tokens": 100
    }' \
    2>/dev/null | jq '.usage.completion_tokens'
done
```

### Profile with profiler
```bash
# Inside container
docker exec vllm-nemotron-single python -m cProfile -s cumtime \
  -m vllm.entrypoints.openai.api_server ...
```

---

## Cleaning Up

```bash
# Stop all benchmarks
docker stop vllm-nemotron-single sglang-nemotron-single
docker stop vllm-nemotron-node1
ssh root@10.0.0.2 docker stop vllm-nemotron-node2

# Remove containers and cleanup
docker rm vllm-nemotron-single sglang-nemotron-single
docker system prune -a -f
```

---

## Framework Comparison Checklist

After running benchmarks, fill in the comparison:

| Metric | vLLM Single | vLLM Multi | SGLang Single | TRT-LLM |
|--------|--|--|--|--|
| **Startup time** | ___ | ___ | ___ | ___ |
| **Model load time** | ___ | ___ | ___ | ___ |
| **Throughput (tok/s)** | ___ | ___ | ___ | ___ |
| **Latency (ms)** | ___ | ___ | ___ | ___ |
| **Memory utilization** | ___ | ___ | ___ | ___ |
| **Stability (crashes)** | ___ | ___ | ___ | ___ |
| **API compatibility** | ✅ | ✅ | ✅ | Custom |

---

## Notes

- All scripts check for HF_TOKEN environment variable
- All use `--ipc=host` and `--shm-size 32g` for multi-GPU communication
- All mount `~/.cache/huggingface` to avoid re-downloading models
- Multi-node scripts assume 10.0.0.1 and 10.0.0.2 network setup (QSFP)
- See `../NEMOTRON_MULTINODE_RESEARCH.md` for deep dive on GB10 issues
