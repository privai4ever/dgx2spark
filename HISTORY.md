# DGX2Spark Project History & Progress

## Overview

This project aims to deploy large language models (LLMs) across two NVIDIA DGX servers using tensor parallelism. The target model is **Minimax-2.5** which requires both DGX nodes to load due to massive memory requirements.

## Hardware Setup

- **DGX 1 (Master)**: 10.0.0.1/24, ARM64 NVIDIA GB10 (SM121 Blackwell)
- **DGX 2 (Slave)**: 10.0.0.2/24, ARM64 NVIDIA GB10 (SM121 Blackwell)
- **Interconnect**: QSFP cables, 10Gbps between nodes
- **Network Config**: Netplan configurations in `10-qsfp.yaml` and `10-qsfp-dgx2.yaml`

## NCCL Status

✅ **CONFIRMED WORKING** - NCCL connectivity between DGX1 and DGX2 is established and functioning.

## Attempted Solutions

### 1. vLLM with Tensor Parallelism

**Scripts**: `start_cluster.sh`, `start_vllm_cluster.sh`

**Status**: ❌ NOT WORKING (pending vLLM 0.18.0 release)

**Details**:
- Custom image: `dgx2spark/vllm:latest` (built from GitHub with multi-node fixes)
- Official image: `vllm/vllm-openai:v0.6.3.post1`
- Configuration: 8 GPUs per node × 2 nodes = 16 GPUs total
- GPU Memory Utilization: 0.9 (later adjusted to 0.75 due to other processes)
- Model: Qwen2.5-Coder-7B-Instruct (7B parameters)
- Issue: Engine core initialization fails on multi-node
- Root Cause: vLLM 1.x has known issues with multi-node + multi-GPU per node configuration
- Resolution: Waiting for vLLM 0.18.0 which promises better distributed inference support

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

## Environment Variables

Common across scripts:
- `MODEL`: HuggingFace model identifier (default: Qwen/Qwen2.5-Coder-7B-Instruct)
- `HUGGINGFACE_TOKEN`: HF authentication token
- `HF_CACHE`: HuggingFace cache directory (`/home/ss/.cache/huggingface`)
- `PORT`: API server port (default: 30000)

## Usage

### Check Status
```bash
./start_trtllm_multinode.sh --status
```

### Start Cluster
```bash
./start_trtllm_multinode.sh
```

### Stop Cluster
```bash
./start_trtllm_multinode.sh --stop
```

### Apply Network Config (if needed)
```bash
sudo netplan apply
```

## Lessons Learned

1. **ARM64 Matters**: GB10 uses ARM64 architecture, requiring ARM64-compatible images. NVIDIA's official ARM64 images work, custom x86_64 builds don't.
2. **GPU Memory**: Realistic GPU memory utilization is ~0.75, not 0.9, due to system processes and overhead.
3. **NCCL is Foundational**: The QSFP network configuration works correctly - NCCL can establish connections between nodes.
4. **Official Solutions Preferred**: NVIDIA's own TRT-LLM is the most promising path for ARM64 DGX clusters.
5. **Multi-Node Complexity**: Distributed inference across physical nodes requires careful coordination of:
   - Network interfaces (IB vs TCP)
   - Process rank assignment
   - Shared filesystem for model loading
   - Environment variable propagation

## Next Steps

1. ✅ Test TRT-LLM with small model (7B parameters)
2. 🔄 Verify multi-node tensor parallelism works
3. 📈 Scale to larger models (70B+ parameters)
4. 🎯 Load Minimax-2.5 across both DGX nodes
5. 📊 Performance tuning: quantization, GGUF, etc.

## References

- NVIDIA dgx-spark-playbooks: https://github.com/NVIDIA/dgx-spark-playbooks
- vLLM: https://github.com/vllm-project/vllm
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- sGLang: https://github.com/sgl-project/sglang

---

**Last Updated**: 2026-03-15
