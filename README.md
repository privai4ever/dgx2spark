# DGX2Spark - Multi-Node LLM Deployment on ARM64 DGX Clusters

**Goal**: Deploy massive LLMs (200GB+) across two NVIDIA DGX GB10 servers using tensor parallelism.

## Hardware

- **DGX 1**: 10.0.0.1, 8×Blackwell GPUs (ARM64 GB10)
- **DGX 2**: 10.0.0.2, 8×Blackwell GPUs (ARM64 GB10)
- **Interconnect**: QSFP 10Gbps
- **Total GPUs**: 16

## Quick Start

```bash
# Recommended: TensorRT-LLM multi-node cluster
./start_trtllm_multinode.sh

# Check status
./start_trtllm_multinode.sh --status

# Stop cluster
./start_trtllm_multinode.sh --stop
```

API endpoint: `http://10.0.0.1:8000/v1/chat/completions`

## Documentation

- **[HISTORY.md](HISTORY.md)** - Complete test log with all approaches tried
- **[METHODOLOGY.md](METHODOLOGY.md)** - Technical comparison of vLLM, sGLang, TensorRT-LLM, etc.
- **[CLAUDE.md](CLAUDE.md)** - Instructions for Claude Code assistant

## Current Status

✅ **TensorRT-LLM multi-node script ready for deployment**

We've tested:
1. ❌ vLLM multi-node (broken, needs v0.18.0)
2. ❌ sGLang multi-node (ARM64 incompatibility)
3. ⚠️ Spark vLLM (replication only, not TP)
4. ✅ **TensorRT-LLM** (official NVIDIA solution, recommended)

See [HISTORY.md](HISTORY.md) for detailed test results.

## Repository Structure

```
├── start_trtllm_multinode.sh   # ⭐ Recommended - TensorRT-LLM cluster
├── start_cluster.sh            # vLLM (when v0.18.0 released)
├── start_sglang.sh             # sGLang (not working)
├── start_spark_vllm.sh         # Spark integration (replication)
├── start_vllm_cluster.sh       # vLLM variant
├── 10-qsfp.yaml                # DGX1 network config
├── 10-qsfp-dgx2.yaml           # DGX2 network config
├── docker-compose.yml
├── Dockerfile.vllm             # Custom vLLM build
├── HISTORY.md                  # Test log & progress
├── METHODOLOGY.md              # Technical comparison
└── CLAUDE.md                   # Assistant instructions
```

## Requirements

- SSH key-based auth between DGX1 and DGX2
- Docker on both nodes
- HuggingFace token for model download
- Netplan network config applied (`sudo netplan apply`)

## Target Models

- **Test**: Qwen2.5-Coder-7B-Instruct (7B)
- **Goal**: Minimax-2.5 (~200GB, requires 16 GPUs with FP8)

---

**Project**: https://github.com/privai4ever/dgx2spark
**Last Updated**: 2026-03-15
