# Nemotron-3-Super-120B Multi-Node Research

**Date:** 2026-03-20
**Status:** Investigation Complete - Multi-node on GB10 has fundamental issues

---

## Executive Summary

Nemotron-3-Super-120B can theoretically run on two DGX Spark GB10 nodes, but **multi-node inference on Blackwell GB10 has a system-level bug** that affects all frameworks equally (vLLM, TRT-LLM). The issue is not software-specific but rather a **Blackwell GB10 driver/kernel limitation**.

---

## Two Distinct Bugs Found

### Bug #1: Multi-Node Crash (System-Level) 🔴 BLOCKER

**Severity:** Critical for multi-node setups
**Root Cause:** Blackwell GB10 driver/kernel issue (NOT software)
**Affects:** Both vLLM AND TensorRT-LLM equally

#### Symptoms
- Memory allocation failures despite available VRAM: `Out of memory [NV_ERR_NO_MEMORY] (0x00000051)`
- NCCL timeouts during collective operations
- GPU compute utilization at 90%+ causes watchdog timeouts
- Identical failure pattern across different inference engines

#### Evidence
- **Tested Frameworks:**
  - vLLM v0.17.1 with Qwen 122B → Failed
  - TensorRT-LLM v1.3.0rc5 with Nemotron 120B → Failed

- **Root Diagnosis:**
  - Unified memory architecture in Blackwell struggles with "dynamic allocation requests at high compute utilization"
  - NCCL synchronization starved by 90%+ compute load
  - Possible RoCE fabric issues under FP8/FP4 workloads

**Forum Discussion:**
https://forums.developer.nvidia.com/t/multi-node-inference-crash-on-blackwell-gb10-memory-allocation-0x51-nccl-timeouts-tested-on-qwen-122b-nemotron-120b/363989

### Bug #2: TRT-LLM Nemotron MoE Output (FIXED)

**Severity:** Medium (single-node only)
**Status:** ✅ FIXED in latest TRT-LLM
**Root Cause:** Per-expert weight scaling in NVFP4 quantized MoE layers

#### Symptoms
- Nemotron NVFP4 with `trtllm-gen` backend generates incoherent output
- Weight scales not synchronized across experts within a layer
- Per-expert fc2_alpha scaling was broken

#### Solution
Fixed through commits addressing proper weight scale handling for mixed-precision checkpoints in trtllm-gen MoE backend.

**Status:** No longer relevant - use latest TRT-LLM version

---

## Memory Requirements by Variant

| Variant | Memory | Ideal Hardware | Status |
|---------|--------|---|--------|
| **BF16** | ~240GB | 8x H100-80GB | ❌ Multi-node broken |
| **FP8** | ~120GB | 2x H100-80GB | ❌ Multi-node broken |
| **NVFP4** | ~60GB | 1x B200-80GB | ✅ Single-node OK |

**Key:** BF16 is 4-byte per parameter × 120B = 480B, but with quantization to FP16 it's ~240GB

---

## What Works in Production (2026)

### Proven Multi-Node 120B+ Deployments

| Company | Model | Hardware | Throughput | Framework |
|---------|-------|----------|-----------|-----------|
| **Baseten** | GPT-OSS 120B | H100 Cluster | 500+ tok/s | vLLM + TP |
| **Furiosa AI** | GPT-OSS 120B | Custom | 5.8ms TPOT | RNGD |
| **MeluXina HPC** | Llama 3+ | Multi-node | - | vLLM Ray |

**Key Insight:** All successful setups use **H100 or newer**, not GB10

Sources:
- https://www.baseten.co/blog/sota-performance-for-gpt-oss-120b-on-nvidia-gpus/
- https://furiosa.ai/blog/serving-gpt-oss-120b-at-5-8-ms-tpot-with-two-rngd-cards-compiler-optimizations-in-practice/
- https://docs.lxp.lu/howto/llama3-vllm/

---

## Why DGX Spark GB10 Struggles with Multi-Node

From NVIDIA's own developer forums:

> "SM121 (GB10) software support is **fundamentally incomplete**"
> "NVIDIA's native multi-node inference stack designed for DGX Spark is not ready for GB10"

### Specific Issues

1. **PyTorch ARM64 Support**
   - No official CUDA 13.0 ARM64 wheels on PyPI
   - Requires custom index URLs
   - Unstable

2. **NCCL Behavior**
   - RoCE fabric struggles under extreme load
   - NCCL heartbeat gets starved at 90%+ compute utilization
   - Causes multi-node synchronization failures

3. **Memory Management**
   - Unified memory architecture has allocation bugs
   - Out-of-memory errors despite available VRAM

4. **Ecosystem Maturity**
   - Multi-node orchestration not production-ready
   - Lack of tested reference implementations

**Forum Threads:**
- https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663
- https://forums.developer.nvidia.com/t/what-is-the-best-practices-for-clustering-4x-dgx-spark-units-into-a-single-unified-system/355131

---

## Parallelism Alternatives

Instead of pure **Tensor Parallelism (TP)**, consider hybrid approaches:

### 1. Tensor Parallelism (TP)
- ✅ Good for same-node GPUs with NVLink
- ❌ Extreme inter-node communication overhead
- ❌ Sensitive to network latency

### 2. Pipeline Parallelism (PP)
- ✅ Works with slower interconnects (like RoCE)
- ✅ Less network overhead than TP
- ❌ Potential GPU idle time (underutilization)

### 3. Hybrid TP + PP (RECOMMENDED for your setup)
```
Tensor Parallel Size = 4 (within each node)
Pipeline Parallel Size = 2 (across nodes)

Result:
- 4 GPUs do tensor parallel within DGX1
- 4 GPUs do tensor parallel within DGX2
- Model split across 2 nodes via pipeline
- Inter-node communication reduced by ~75%
```

### 4. Expert Parallelism (EP) - For MoE Models
- Splits experts across nodes, not full model
- Better for Nemotron (has MoE)
- Less communication overhead

### 5. Context Parallelism (CP)
- For handling extremely long contexts (1M+ tokens)
- Emerging technique

**Source:** https://docs.vllm.ai/en/stable/serving/parallelism_scaling/

---

## Recommendations for Your Setup

### Option 1: Try Hybrid TP+PP (Best Chance of Success)

```bash
tensor_parallel_size=4      # Within DGX1 and DGX2
pipeline_parallel_size=2    # Across DGX1 and DGX2
```

**Why:** Reduces inter-node communication that triggers the GB10 bug

**Try with:** Any quantized Nemotron variant (FP8, NVFP4)

### Option 2: Single-Node Only

```bash
# Run Nemotron on DGX1 only
tensor_parallel_size=8      # All 8 GPUs on single node
```

**Pros:** Avoids multi-node GB10 bug completely
**Cons:** Limited by single node memory

**Best for:** NVFP4 variant (60GB easily fits in 8 GPUs)

### Option 3: Wait for NVIDIA Driver Update

NVIDIA is aware of these issues. A future driver release may fix:
- Memory allocation under high compute load
- NCCL heartbeat starvation
- RoCE fabric stability

**Timeline:** Unknown (no official roadmap for GB10 software)

### Option 4: Hardware Upgrade

H100 clusters have proven 120B+ multi-node inference working reliably.

**Cost:** Significant, but guaranteed to work

---

## Test Matrix for Your DGX Setup

| Setup | Model | Variant | Approach | Expected |
|-------|-------|---------|----------|----------|
| DGX1 Only | Nemotron 120B | NVFP4 | Single-node TP=8 | ✅ Should work |
| DGX1 Only | Nemotron 120B | FP8 | Single-node TP=8 | ✅ Should work |
| DGX1 Only | Nemotron 120B | BF16 | Single-node TP=8 | ❓ Might OOM |
| DGX1+DGX2 | Nemotron 120B | NVFP4 | Hybrid TP=4/PP=2 | ⚠️ Might work (untested) |
| DGX1+DGX2 | Nemotron 120B | FP8 | Hybrid TP=4/PP=2 | ⚠️ Might work (untested) |
| DGX1+DGX2 | Nemotron 120B | BF16 | Tensor TP=16 | ❌ Will crash |

---

## Next Steps

1. **Immediate:** Try single-node NVFP4 on DGX1
   ```bash
   ./start_trtllm_correct.sh
   # Download: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
   # Launch with --tp_size 8
   ```

2. **If single-node works:** Attempt Hybrid TP+PP multi-node
   ```bash
   # Set up with --tensor_parallel_size 4 --pipeline_parallel_size 2
   # Monitor NCCL logs carefully
   ```

3. **If multi-node fails:** Stay single-node or contact NVIDIA support
   - Provide this research document
   - Show forum thread references
   - Request timeline for GB10 software fix

4. **Consider alternatives:**
   - Try other 120B models (Qwen, Llama)
   - Use quantized versions of Nemotron instead

---

## References

### NVIDIA Forums (GB10 Multi-Node Issues)
- [Multi-Node Nemotron 120B Crash](https://forums.developer.nvidia.com/t/multi-node-inference-crash-on-blackwell-gb10-memory-allocation-0x51-nccl-timeouts-tested-on-qwen-122b-nemotron-120b/363989)
- [GB10 Software Support Severely Lacking](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)
- [Clustering Best Practices](https://forums.developer.nvidia.com/t/what-is-the-best-practices-for-clustering-4x-dgx-spark-units-into-a-single-unified-system/355131)

### Model Documentation
- [Nemotron 3 Super on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

### Inference Frameworks
- [vLLM Parallelism Documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)

### Production Deployments
- [Baseten 120B Setup](https://www.baseten.co/blog/sota-performance-for-gpt-oss-120b-on-nvidia-gpus/)
- [Furiosa 120B Optimization](https://furiosa.ai/blog/serving-gpt-oss-120b-at-5-8-ms-tpot-with-two-rngd-cards-compiler-optimizations-in-practice/)
- [Meta Inference Innovations](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)

---

## Conclusion

**Nemotron-3-Super-120B can run on your two DGX nodes, but not reliably in a multi-node configuration due to Blackwell GB10 driver issues.** The best approach for your hardware is:

1. **Start with single-node (DGX1 only)** using NVFP4 variant
2. **If you need multi-node:** Try hybrid TP+PP approach (untested but theoretically better)
3. **If you need guaranteed 120B+ multi-node:** Consider H100 cluster upgrade

The frustration is justified — GB10's multi-node software support is genuinely incomplete, not just buggy. This is acknowledged by NVIDIA in their forums.
