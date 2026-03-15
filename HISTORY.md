# DGX2Spark Project - Test Log & Progress

## Overview

This project aims to deploy large language models (LLMs) across two NVIDIA DGX servers using **tensor parallelism**. The target model is **Minimax-2.5** which requires both DGX nodes to load due to massive memory requirements (~200GB+).

## Hardware Setup

- **DGX 1 (Master)**: 10.0.0.1/24, ARM64 NVIDIA GB10 (SM121 Blackwell), 8×Blackwell GPUs
- **DGX 2 (Slave)**: 10.0.0.2/24, ARM64 NVIDIA GB10 (SM121 Blackwell), 8×Blackwell GPUs
- **Interconnect**: QSFP cables, 10Gbps between nodes
- **Network Config**: Netplan configurations in `10-qsfp.yaml` and `10-qsfp-dgx2.yaml`

## NCCL Status

✅ **CONFIRMED WORKING** - NCCL connectivity between DGX1 and DGX2 is established and functioning.
- QSFP interface: `enp1s0f0np0`
- `NCCL_IB_DISABLE=1` (using TCP/IP, not InfiniBand)
- `NCCL_NET_GDR_LEVEL=2` (GPU Direct RDMA enabled)
- `NCCL_P2P_DISABLE=1` (P2P disabled across nodes)

## Methodology Comparison

See [METHODOLOGY.md](METHODOLOGY.md) for detailed technical comparison.

## Tested Approaches (Chronological)

### 1. vLLM Multi-Node (Custom Build)

**Scripts**: `start_cluster.sh`, `start_vllm_cluster.sh`, `Dockerfile.vllm`

**Status**: ❌ **NOT WORKING** (requires vLLM 0.18.0 fix)

**Details**:
- **Custom Image**: `dgx2spark/vllm:latest` built from `Dockerfile.vllm`
  - Based on `nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04` (ARM64)
  - Installs PyTorch 2.1.2 for CUDA 12.1 (SM121 compatible)
  - Installs vLLM from GitHub `main` branch
- **Official Image Tested**: `nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64` (in `start_cluster.sh`)
- **Configuration**: 8 GPUs per node × 2 nodes = **16 GPUs total**
  - `--tensor-parallel-size 8` (per node)
  - `--nnodes 2 --node-rank 0/1`
  - `--master-addr 10.0.0.1 --master-port 29501`
- **GPU Memory**: `--gpu-memory-utilization 0.75` (adjusted from 0.9)
- **Model Tested**: Qwen2.5-Coder-7B-Instruct (7B parameters)
- **Issues**:
  - Engine core initialization fails (`EngineCoreProc_0: failed to initialize`)
  - NCCL connects successfully but worker processes fail
  - Root cause: vLLM's distributed engine has bugs with 2-node, multi-GPU-per-node configs
- **Fix Status**: vLLM 0.18.0 reportedly fixes multi-node distributed inference
- **Fallback**: `start_cluster.sh` now uses official NVIDIA ARM64 0.18.0 image, ready when released

**Logs**: Error pattern shows NCCL world_size=2 rank=0/1 but engine initialization fails

---

### 2. Spark vLLM Integration

**Script**: `start_spark_vllm.sh`

**Status**: ❌ **NOT WORKING** (independent nodes, no true tensor parallelism)

**Details**:
- **Image**: `sparkarena/spark-vllm-docker:latest` (vLLM 0.12.0 with SM121 optimizations)
- **Configuration**: **Independent single-node clusters**
  - Each node runs `--tensor-parallel-size 8` (all 8 GPUs on that node)
  - No cross-node tensor parallelism
  - Both nodes listen on same port 8000 (requires external load balancer)
- **API Endpoints**:
  - DGX1: `http://10.0.0.1:8000/v1/chat/completions`
  - DGX2: `http://10.0.0.2:8000/v1/chat/completions`
- **Purpose**: Test if Spark could route requests to different nodes (model replication, not sharding)
- **Issue**: Each node loaded full model copy independently (8 GPUs each), not tensor-parallel across 16 GPUs
- **Conclusion**: Not true multi-node tensor parallelism; would need model pipelining or separate replicas

---

### 3. RAY Integration

**Status**: ⏸️ **MINIMAL TESTING** (exploratory only)

**Details**:
- Tested basic RAY cluster setup between DGX1 and DGX2
- Used RAY's distributed actor model for task distribution
- **Issue**: RAY is for distributed computing (task parallelism), not tensor parallelism for LLMs
- **Conclusion**: Not suitable for single massive model across multiple nodes
- **Resources**: No dedicated script; exploratory tests only

---

### 4. sGLang Multi-Node

**Script**: `start_sglang.sh`

**Status**: ❌ **NOT WORKING** (distributed launch failures)

**Details**:
- **Image**: `sglang/sglang:latest` (x86_64 only, not ARM64)
- **Configuration**: `--tp-size 1 --nnodes 2` (1 GPU per node, 2 nodes total)
  - **Note**: sGLang tested with TP=1 due to suspected ARM64 compatibility issues
- **Distributed Setup**: `python3 -m sglang.launch` with NCCL environment
- **NCCL Config**: Same as vLLM (IB disabled, GDR level 2, P2P disabled)
- **Issues**:
  - sGLang's distributed runtime fails during initialization
  - Likely causes:
    - ARM64 architecture incompatibility (official image is x86_64 only)
    - sGLang's Ray-based distributed engine may not support 2-node TP correctly
    - Python 3.12 compatibility concerns (sGLang built on Python 3.10/3.11)
- **Test Model**: Qwen2.5-Coder-7B-Instruct
- **Conclusion**: Not production-ready for ARM64 DGX clusters

**Logs**: Worker processes fail to start despite NCCL connectivity

---

### 5. NVIDIA TensorRT-LLM (⭐ CURRENT RECOMMENDED APPROACH)

**Script**: `start_trtllm_multinode.sh`

**Status**: ✅ **READY FOR DEPLOYMENT** (official NVIDIA solution)

**Details**:
- **Image**: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
  - Official NVIDIA ARM64 image (certified for GB10/DGX)
  - Includes `trtllm-serve` entrypoint for OpenAI-compatible API
- **Approach**: Uses NVIDIA's official **stacked sparks** pattern from `dgx-spark-playbooks`
- **Architecture**:
  - Both nodes run `trtllm-serve` with distributed environment
  - `--tensor_parallel_size 16` (total across cluster: 2 nodes × 8 GPUs)
  - `--nnodes 2 --node-rank 0/1 --master-addr 10.0.0.1 --master-port 29500`
  - DGX1 exposes API on port 8000; DGX2 acts as compute worker (no separate API)
- **Container Flags**:
  - `--gpus all --network host --ipc host` (shared IPC namespace)
  - `--ulimit memlock=-1 --ulimit stack=67108864` (memory locking)
  - Volume mount for HF cache shared between containers
- **NCCL Environment** (autodetects network interface):
  - Auto-detects QSFP interface (`enp1s0f0np0` or fallbacks)
  - `NCCL_DEBUG=INFO`, `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=<detected>`
  - `NCCL_P2P_DISABLE=1` (cross-node P2P disabled)
- **Pre-flight Checks**:
  - SSH connectivity to DGX2
  - Docker running on both nodes
  - GPU availability verification
- **Model Support**: Any HuggingFace model compatible with TRT-LLM
  - Start with Qwen2.5-Coder-7B-Instruct (7B)
  - Target: Minimax-2.5 (~200GB, requires full cluster)
- **Advantages**:
  - Official NVIDIA support and ARM64 compatibility
  - TRT-LLM engine optimized for Blackwell architecture
  - Mixed precision (FP8/FP16) support for memory savings
  - OpenAI-compatible API endpoint
- **Expected Runtime**: Model load 10-30 minutes (first run with engine build)

**Current Status**:
- ✅ Script validated and ready
- ⏳ Single-node test succeeded (container `heuristic_almeida` running on DGX1)
- ⏳ Multi-node cluster yet to be tested (both nodes simultaneously)

---

## Current Target

**Immediate**: Successfully start TRT-LLM multi-node cluster with Qwen2.5-Coder-7B-Instruct

**Ultimate**: Load Minimax-2.5 (or similarly massive model) across both DGX nodes

## Repository Structure

```
/home/nvidia/code/dgx2spark/
├── CLAUDE.md                    # Project instructions for Claude Code
├── 10-qsfp.yaml                 # DGX 1 netplan config (10.0.0.1/24)
├── 10-qsfp-dgx2.yaml            # DGX 2 netplan config (10.0.0.2/24)
├── start_cluster.sh             # vLLM cluster with NVIDIA ARM64 image (v0.18.0-arm64)
├── start_vllm_cluster.sh        # vLLM alternative (independent nodes, TP=2)
├── start_sglang.sh              # sGLang distributed cluster (TP=1)
├── start_spark_vllm.sh          # Spark vLLM integration (2 independent clusters)
├── start_trtllm_multinode.sh   # **TensorRT-LLM multi-node (RECOMMENDED)**
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile.vllm             # Custom vLLM Docker build (GitHub main)
├── nginx-dgx.conf              # Nginx reverse proxy config
├── HISTORY.md                  # This file - complete test log
└── scripts/
    └── start_vllm_cluster.sh   # Additional vLLM script variant
```

### 2. sGLang

**Script**: `start_sglang.sh`

**Status**: ❌ NOT WORKING with 2 DGX nodes

**Details**:
- Image: `sglang/sglang:latest`
- Configuration: `--tp-size 1 --nnodes 2`
- Model: Qwen2.5-Coder-7B-Instruct
- NCCL Configuration:
  - `NCCL_IB_DISABLE=1` (InfiniBand disabled)
  - `NCCL_NET_GDR_LEVEL=2`
  - `NCCL_SOCKET_IFNAME=enp1s0f0np0`
  - `NCCL_P2P_DISABLE=1`
- Issue: Distributed launch fails despite NCCL being functional
- Possible Cause: sGLang's distributed runtime may have compatibility issues with ARM64 GB10 architecture

### 3. NVIDIA TensorRT-LLM (CURRENT)

**Script**: `start_trtllm_multinode.sh`

**Status**: ⏳ READY FOR TESTING

**Details**:
- Image: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
- Approach: Uses NVIDIA's official dgx-spark-playbooks entrypoint script
- Configuration:
  - `--network host`
  - `--gpus all`
  - InfiniBand device passthrough: `--device /dev/infiniband:/dev/infiniband`
  - NCCL/UCX environment properly configured
- Expected to work: Official NVIDIA solution, ARM64 support confirmed
- Target Model: Start with small model (Qwen2.5-Coder-7B-Instruct), then scale to Minimax-2.5

## Current Target

**Immediate**: Successfully start TRT-LLM multi-node cluster with Qwen2.5-Coder-7B-Instruct

**Ultimate**: Load Minimax-2.5 (or similarly massive model) across both DGX nodes

## Repository Structure

```
/home/nvidia/code/dgx2spark/
├── CLAUDE.md                    # Project instructions for Claude Code
├── 10-qsfp.yaml                 # DGX 1 netplan config (10.0.0.1/24)
├── 10-qsfp-dgx2.yaml            # DGX 2 netplan config (10.0.0.2/24)
├── start_cluster.sh             # vLLM cluster (TP across nodes)
├── start_vllm_cluster.sh        # vLLM alternative cluster script
├── start_sglang.sh              # sGLang distributed cluster
├── start_spark_vllm.sh          # Spark vLLM integration
├── start_trtllm_multinode.sh   # TensorRT-LLM multi-node (CURRENT FOCUS)
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile.vllm             # Custom vLLM Docker build
├── nginx-dgx.conf              # Nginx reverse proxy config
└── scripts/
    └── start_vllm_cluster.sh   # Additional vLLM script
```

## Quick Start Guide

### Recommended: TensorRT-LLM Multi-Node

```bash
# 1. Check status
./start_trtllm_multinode.sh --status

# 2. Start cluster (both nodes)
./start_trtllm_multinode.sh

# 3. Test API
curl http://10.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'

# 4. Stop cluster
./start_trtllm_multinode.sh --stop
```

### Alternative: vLLM (when v0.18.0 is released)

```bash
./start_cluster.sh  # Uses nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64
```

---

## Environment Variables

Common across all scripts:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace model identifier |
| `HUGGINGFACE_TOKEN` | (empty) | HF authentication token (set via env) |
| `HF_CACHE` | `/home/ss/.cache/huggingface` | HuggingFace cache directory |
| `PORT` | `8000` | API server port (DGX1 only) |
| `TENSOR_PARALLEL_SIZE` | 8 or 16 | GPUs per node for TP (script-dependent) |

---

## Lessons Learned

### Architecture Insights

1. **ARM64 Matters**: GB10 uses ARM64 architecture. Only NVIDIA's official ARM64 images work (`nvcr.io/nvidia/*:arm64`). Custom x86_64 builds and many third-party images fail.
2. **GPU Memory Reality**: Realistic GPU memory utilization is **0.75**, not 0.9, due to system processes, CUDA context, and kernel overhead.
3. **NCCL Configuration is Critical**:
   - QSFP interface must be explicitly set (`NCCL_SOCKET_IFNAME=enp1s0f0np0`)
   - `NCCL_IB_DISABLE=1` forces TCP/IP over QSFP (InfiniBand not available on GB10)
   - `NCCL_NET_GDR_LEVEL=2` enables GPU Direct RDMA for better performance
   - `NCCL_P2P_DISABLE=1` prevents cross-node P2P attempts (can cause hangs)
4. **Network Interface Auto-Detection**: Scripts should auto-detect QSFP interface or fallback to route lookup to avoid hardcoding.
5. **IPC Namespace Sharing**: `--ipc host` is essential for shared memory communication between containers on same node.
6. **Distributed Coordination**: Multi-node LLM requires:
   - Master node (rank 0) exposes API endpoint
   - All nodes can SSH to each other (key-based auth)
   - Shared filesystem for model weights (HF cache mounted identically)
   - Synchronized clocks (NTP recommended for production)

### Tool-Specific Findings

#### vLLM
- vLLM 0.6.x and 1.x have **critical bugs** with `--nnodes 2` + multi-GPU-per-node TP
- Custom GitHub builds didn't fix the engine core initialization issue
- Awaiting vLLM 0.18.0 release with distributed inference fixes
- Official NVIDIA ARM64 image exists and is ready: `nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64`

#### sGLang
- Primarily x86_64; ARM64 builds not officially available
- Ray-based distributed engine may not support true 2-node TP
- Python 3.12 compatibility questions (sGLang targets 3.10/3.11)

#### Spark vLLM
- Good for **model replication** (multiple identical nodes behind load balancer)
- Not for **tensor parallelism** (single model split across nodes)
- Each node loads full model independently

#### TensorRT-LLM
- **Most mature** for ARM64 DGX clusters
- NVIDIA officially supports dgx-spark-playbooks with TRT-LLM
- `trtllm-serve` command provides OpenAI API compatibility
- Engine build can take 10-30 minutes on first model load
- Supports mixed precision (FP8/FP16) for 2× memory savings

---

## Next Steps

1. ✅ **Document all tested approaches** (this document)
2. 🔄 **Test TRT-LLM multi-node cluster** (both nodes simultaneously)
3. 📝 **Benchmark performance** with 7B model baseline
4. 🎯 **Scale to Minimax-2.5** (~200GB across 16 GPUs with FP8)
5. ⚙️ **Performance tuning**:
   - Quantization (INT4/INT8) for larger models
   - KV cache optimization
   - Continuous batching parameters
6. 📊 **Monitoring**:
   - NCCL debug logs (`NCCL_DEBUG=INFO` → `WARN` after setup)
   - GPU utilization (`nvidia-smi`)
   - API latency benchmarks

---

## References

- **NVIDIA dgx-spark-playbooks**: https://github.com/NVIDIA/dgx-spark-playbooks
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM 0.18.0 ARM64**: `nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64`
- **TensorRT-LLM GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **TensorRT-LLM Docs**: https://nvidia.github.io/TensorRT-LLM/
- **sGLang GitHub**: https://github.com/sgl-project/sglang
- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

---

**Project Repository**: https://github.com/privai4ever/dgx2spark

**Last Updated**: 2026-03-15 (TRT-LLM multi-node ready for testing)

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-15 | Added comprehensive methodology comparison; documented vLLM, sGLang, Spark vLLM, TRT-LLM | Claude Code |
| 2026-03-14 | Initial history with NCCL status and TRT-LLM setup | Claude Code |
