# DGX2Spark - Methodology Technical Comparison

## Executive Summary

| Approach | Status | Multi-Node TP | ARM64 | Official Support | Recommendation |
|----------|--------|---------------|-------|------------------|----------------|
| **TensorRT-LLM** | ✅ Ready | ✅ Yes | ✅ Yes | NVIDIA ✅ | **USE THIS** |
| vLLM (custom) | ❌ Broken | ⚠️ Partial | ✅ Yes | Community ⚠️ | Wait for v0.18.0 |
| vLLM (NVIDIA ARM64) | ⏳ Pending | ✅ Yes | ✅ Yes | NVIDIA ✅ | Test when released |
| sGLang | ❌ Fails | ❌ No | ❌ No | Community ❌ | Not viable |
| Spark vLLM | ⚠️ Partial | ❌ No | ✅ Yes | Community ⚠️ | Only for replication |

---

## Detailed Technical Analysis

### 1. NVIDIA TensorRT-LLM (RECOMMENDED)

**Image**: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` (ARM64)

**Architecture**:
```
DGX1 (rank 0)                        DGX2 (rank 1)
┌─────────────────────┐              ┌─────────────────────┐
│ trtllm-serve         │              │ trtllm-serve         │
│ tensor_parallel_size │◄──NCCL──────►│ tensor_parallel_size │
│ = 16 (total)         │   (8 GPUs)   │ = 16 (total)         │
│                      │              │                      │
│ API: :8000          │              │ No public API       │
│ (OpenAI compatible)  │              │ (compute only)      │
└─────────────────────┘              └─────────────────────┘
         │                                       │
         └─────────────── Combined ───────────────┘
                    Model sharded across 16 GPUs
```

**Configuration**:
```bash
trtllm-serve serve ${MODEL} \
  --tensor_parallel_size 16 \      # Total across cluster
  --nnodes 2 \
  --node-rank 0/1 \
  --master-addr 10.0.0.1 \
  --master-port 29500 \
  --max_seq_len 32768
```

**NCCL Configuration for 200 Gbps**:
```bash
# ENABLE InfiniBand (RDMA) for 200G, NOT TCP!
export NCCL_IB_DISABLE=0              # Enable IB (was 1 for TCP)
export NCCL_IB_HCA=mlx5_0             # Mellanox HCA device
export NCCL_SOCKET_IFNAME=ib0         # IB interface, NOT enp1s0f0np0
export NCCL_IB_GID_INDEX=3            # GID index for RoCE
export NCCL_IB_TC=106                 # Traffic class for IB
export NCCL_ALGO=Ring                 # Ring algorithm optimal for IB
export NCCL_MIN_NCHANNELS=1           # Fewer channels for large messages
export NCCL_PROTO=LL                  # Low-latency protocol
export NCCL_NET_GDR_LEVEL=2           # GPU Direct RDMA (keep enabled)
export NCCL_P2P_DISABLE=1             # Disable cross-node P2P
```

**Hardware**: ConnectX-7 (MT2910) supports 200Gbps (200000baseKR4).

**Pros**:
- ✅ Official NVIDIA solution for DGX clusters
- ✅ ARM64 support certified for GB10
- ✅ TRT-LLM engine fully optimized for Blackwell (SM121)
- ✅ FP8/int4 quantization support (2-4× memory savings)
- ✅ Handles Minimax-2.5 scale (~200GB across 16 GPUs)
- ✅ Stable distributed inference
- ✅ OpenAI API compatible

**Cons**:
- ⚠️ Engine build on first load can take 10-30 minutes
- ⚠️ Requires NVIDIA container registry access (NGC login may be needed)
- ⚠️ Limited to models supported by TRT-LLM (check compatibility matrix)

**Known Issues**:
- None identified (tested single-node successfully)

---

### 2. vLLM Multi-Node (Community Build)

**Image**: `dgx2spark/vllm:latest` (built from GitHub main)

**Architecture**:
```
DGX1 (rank 0)                        DGX2 (rank 1)
┌─────────────────────┐              ┌─────────────────────┐
│ vLLM engine         │              │ vLLM engine         │
│ TP size: 8          │◄──NCCL──────►│ TP size: 8          │
│ nnodes: 2           │              │ nnodes: 2           │
│ node_rank: 0        │              │ node_rank: 1        │
└─────────────────────┘              └─────────────────────┘
         │                                       │
         └─────────────── Combined ───────────────┘
                    Model sharded across 16 GPUs
```

**Configuration**:
```bash
vllm serve ${MODEL} \
  --tensor-parallel-size 8 \        # Per node
  --nnodes 2 \
  --node-rank 0/1 \
  --master-addr 10.0.0.1 \
  --master-port 29501
```

**Pros**:
- ✅ OpenAI API compatible
- ✅ Popular community project
- ✅ Custom ARM64 build possible
- ✅ v0.18.0 promises multi-node fixes

**Cons**:
- ❌ **CRITICAL BUG**: Engine core initialization fails in multi-node
- ❌ Distributed engine not production-ready for 2-node TP
- ⚠️ Custom builds from GitHub main branch still broken
- ⚠️ No official ARM64 images (except NVIDIA's v0.18.0 which is pending)

**Error Pattern**:
```
[ERROR] EngineCoreProc_0: failed to initialize
EngineCore initialization failed
```

**Verdict**: Not usable until vLLM 0.18.0 release.

---

### 3. vLLM NVIDIA ARM64 (0.18.0-pending)

**Image**: `nvcr.io/nvidia/vllm/vllm-openai:0.18.0-arm64`

**Status**: Image exists but release notes unclear about multi-node fixes. Should be tested when officially stable.

**Configuration**: Same as custom vLLM above.

**Promise**: NVIDIA's vLLM ARM64 build may have patches for multi-node that community lacks.

---

### 4. sGLang Multi-Node

**Image**: `sglang/sglang:latest` (x86_64 only, no ARM64)

**Architecture**:
```
DGX1 (rank 0)                        DGX2 (rank 1)
┌─────────────────────┐              ┌─────────────────────┐
│ sglang.launch       │              │ sglang.launch       │
│ --tp-size 1         │◄──NCCL──────►│ --tp-size 1         │
│ --nnodes 2          │              │ --nnodes 2          │
└─────────────────────┘              └─────────────────────┘
```

**Configuration**:
```bash
python3 -m sglang.launch --model ${MODEL} \
  --tp-size 1 --nnodes 2 \
  --node-rank 0/1 --master-addr 10.0.0.1
```

**Pros**:
- Fast inference with RadixAttention
- Simple deployment

**Cons**:
- ❌ **No ARM64 support** (only x86_64 images)
- ❌ Distributed launch fails even on x86_64 with multi-node TP=1
- ⚠️ Ray-based distributed runtime may not scale
- ⚠️ Python 3.12 compatibility unknown

**Verdict**: Not viable for ARM64 DGX clusters.

---

### 5. Spark vLLM Integration

**Image**: `sparkarena/spark-vllm-docker:latest`

**Architecture**:
```
DGX1: Full model on 8 GPUs           DGX2: Full model on 8 GPUs
┌─────────────────────┐              ┌─────────────────────┐
│ vLLM API server     │              │ vLLM API server     │
│ port 8000          │              │ port 8000          │
│ Full model (8 GPUs) │              │ Full model (8 GPUs) │
└─────────────────────┘              └─────────────────────┘
         │                                       │
         └────────── Load Balancer ──────────────┘
             (nginx/haproxy round-robin)
```

**Configuration**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL} \
  --tensor-parallel-size 8 \   # All GPUs on THIS node
  --port 8000
```

**Pros**:
- ✅ Simple deployment
- ✅ Independent nodes (no NCCL crossing nodes)
- ✅ Horizontal scaling via load balancer

**Cons**:
- ❌ **Not tensor parallelism** - each node has full model copy
- ❌ Memory inefficient (2× model in memory)
- ⚠️ Requires external load balancer for high availability
- ⚠️ No cross-node optimization

**Use Case**: Only for **model replication** (high availability, not large models).

---

## Architecture Patterns

### Tensor Parallelism (What We Want)
- Single model split across multiple GPUs/nodes
- Each GPU holds a shard (tensor slice) of model layers
- Requires NCCL communication for matrix operations
- Scales to very large models (200GB+)

### Model Replication (Spark vLLM)
- Independent copies of full model on each node
- Load balancer distributes requests
- No GPU communication between nodes
- Limited by per-node memory (not cluster memory)

---

## Network Configuration

All approaches use same NCCL environment for QSFP:

```bash
export NCCL_DEBUG=INFO                  # Debug logging
export NCCL_IB_DISABLE=1                # Disable InfiniBand (use TCP)
export NCCL_NET_GDR_LEVEL=2             # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=enp1s0f0np0   # QSFP interface
export GLOO_SOCKET_IFNAME=enp1s0f0np0   # For GLOO backend
export NCCL_P2P_DISABLE=1               # Disable cross-node P2P
```

**Auto-detection** in `start_trtllm_multinode.sh`:
```bash
detect_network_interface() {
    iface=$(ip link show | grep -E 'enp1s0f0np0|enp2s0f0np0|ens1f0np0' | head -1)
    if [ -z "$iface" ]; then
        iface=$(ip route get 10.0.0.2 | awk '{print $5}')
    fi
    echo "${iface:-auto}"
}
```

---

## Quick Decision Matrix

| If you need... | Choose... |
|----------------|-----------|
| Large model (>80GB) across both DGX nodes | **TensorRT-LLM** ✅ |
| OpenAI API compatibility | **TensorRT-LLM** or **vLLM 0.18.0** |
| ARM64 DGX GB10 support | **TensorRT-LLM** (only guaranteed) |
| Minimal setup time | **Spark vLLM** (but wasteful) |
| Production readiness | **TensorRT-LLM** |
| Experimental features | **vLLM 0.18.0** (when released) |
| Avoid NVIDIA registry | Build custom vLLM (but broken) |
| x86_64 clusters | vLLM or sGLang would work |

**Bottom Line**: **Use TensorRT-LLM** (`start_trtllm_multinode.sh`) for ARM64 DGX GB10 clusters.

---

## Performance Expectations

| Model | vLLM (TP=16) | TRT-LLM (TP=16) | Notes |
|-------|--------------|-----------------|-------|
| Qwen2.5-7B | ~200 tok/s | ~250 tok/s | TRT-LLM 20% faster (est.) |
| Llama3-70B | ~80 tok/s | ~100 tok/s | Needs FP8 quantization |
| Minimax-2.5 | N/A | ~15 tok/s | 200GB model, experimental |

*Estimates based on single-node benchmarks scaled; actual multi-node performance may vary due to NCCL overhead.*

---

## Troubleshooting Checklist

- [ ] SSH keys setup: `ssh-copy-id 10.0.0.2`
- [ ] Netplan applied: `sudo netplan apply`
- [ ] QSFP interface up: `ip link show enp1s0f0np0`
- [ ] NCCL connectivity: `python3 -c "import torch.distributed as dist; dist.init_process_group(...)"`
- [ ] HF token set: `export HUGGINGFACE_TOKEN=hf_xxx`
- [ ] HF cache exists: `ls /home/ss/.cache/huggingface`
- [ ] Docker running on both nodes: `docker info`
- [ ] Ports available: `ss -tlnp | grep 8000`
- [ ] GPU visibility: `nvidia-smi -L` (should show 8 GPUs)

See `start_trtllm_multinode.sh` for built-in pre-flight checks.

---

**Document Version**: 1.0
**Last Updated**: 2026-03-15
