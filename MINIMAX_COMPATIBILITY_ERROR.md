# MiniMax-M2.5 Compatibility Error Report

## Summary
MiniMax-M2.5 cannot be loaded with TensorRT-LLM 1.0.0rc3 due to unsupported model architecture.

## Error Details

**Error Message:**
```
ValueError: Unknown architecture for AutoModelForCausalLM: MiniMaxM2ForCausalLM
```

**Location:**
- File: `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_auto.py`, line 26
- Framework: TensorRT-LLM 1.0.0rc3

## Root Cause

The MiniMax-M2.5 model uses a custom architecture type `MiniMaxM2ForCausalLM` which is **not registered** in TensorRT-LLM's AutoModelForCausalLM registry for the `pytorch` backend.

When TRT-LLM tries to instantiate the model via:
```python
AutoModelForCausalLM.from_config(config)
```

The framework cannot find a handler for this architecture type and raises `ValueError: Unknown architecture`.

## What Was Tested

| Model | Size | TRT-LLM 1.0.0rc3 | Multi-Node | Status |
|-------|------|------------------|-----------|--------|
| Llama-3.1-8B | 2.6GB | ✅ Supported | ✅ Works | **Functional** |
| Qwen2.5-Coder-7B | 15GB | ✅ Supported | ✅ Works | **Functional** |
| Qwen3-235B | 12GB (FP4) | ✅ Supported | ⏱️ Timeout | Kernel compilation too slow |
| **MiniMax-M2.5** | **198GB** | ❌ **Unsupported** | ❌ Failed | **Incompatible** |

## Why MiniMax Failed

1. **Custom Architecture:** MiniMax uses `MiniMaxM2ForCausalLM` architecture, not a standard LLaMA/Qwen variant
2. **Framework Limitation:** TensorRT-LLM 1.0.0rc3 only supports specific registered architectures
3. **No Fallback:** The `pytorch` backend doesn't have a generic fallback for unknown architectures
4. **Not a Model Sync Issue:** Unlike earlier attempts, this is NOT due to missing files on DGX2 - it's a framework-level incompatibility

## Test Scenario

```bash
# Configuration that failed
MODEL="MiniMaxAI/MiniMax-M2.5"
TP_SIZE=2  # Multi-node (1 per DGX)
BACKEND="pytorch"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3"

# Command
docker exec trtllm-multinode bash -c "
  mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve MiniMaxAI/MiniMax-M2.5 \
    --tp_size 2 \
    --backend pytorch \
    --port 8355
"

# Result: ❌ Fails during model initialization
```

## Workaround

There is **no workaround** with TensorRT-LLM 1.0.0rc3. Options:

1. **Use compatible models** (recommended):
   - Llama-3.1-8B (proven multi-node working)
   - Qwen2.5-Coder-7B (proven multi-node working)

2. **Try newer TRT-LLM versions** (if available):
   - Version 1.2.0+ may have added MiniMax support
   - However, 1.2.0rc6 has SM121 Blackwell GPU issues (see NVIDIA issue #8474)

3. **Use alternative inference framework**:
   - vLLM: Check if vLLM has MiniMax architecture support
   - Ollama: If MiniMax weights available in GGUF format

## Community Context: Wider TRT-LLM Architecture Support Issues

This is **not an isolated issue**. The TensorRT-LLM project has a pattern of architecture-specific incompatibilities:

### Known Similar Issues

- **Falcon Model (FalconForCausalLM):** Reported unsupported in [GitHub Issue #1116](https://github.com/NVIDIA/TensorRT-LLM/issues/1116)
- **Gemma-3 Loading:** Issues loading Gemma-3 after certain commits affecting model detection ([Issue #6193](https://github.com/NVIDIA/TensorRT-LLM/issues/6193))
- **Thor/SM101 Support:** Requested but not implemented with INT4/INT8 quantization ([Issue #5594](https://github.com/NVIDIA/TensorRT-LLM/issues/5594))
- **DGX Spark (SM121) General Issues:** Multiple bugs reported:
  - FP4 CUTLASS GEMM fails on GB10 due to shared memory overflow from B200-sized configs ([Issue #11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368))
  - Multi-node inference stack not ready for GB10/SM121 ([Issue #8474](https://github.com/NVIDIA/TensorRT-LLM/issues/8474))

### NVIDIA's Own Documentation

NVIDIA's [TensorRT-LLM Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) lists model architectures, but version-specific compatibility is not always clear. MiniMax-M2 is mentioned in NVIDIA documentation, but no version/GPU-specific compatibility information is provided.

### The Pattern

1. **Model architectures added gradually** - not all models supported in all versions
2. **GPU architecture constraints** - SM121 (Blackwell/DGX Spark) has specific limitations
3. **Version fragmentation** - Support varies between 1.0.0, 1.0.0rc3, 1.2.0rc6, etc.
4. **Documentation gap** - NVIDIA docs don't clearly specify which models work on which GPU architectures for which versions

### Why DGX Buyers Are Affected

DGX Spark customers purchasing for specific model deployment (e.g., MiniMax, Qwen3-235B) may find:
- ❌ Models listed in NVIDIA docs as "supported"
- ❌ But incompatible with available/stable TRT-LLM versions
- ❌ And alternative frameworks (vLLM) also have SM121 issues
- ➡️ **Result:** Expensive hardware cannot run desired models efficiently

This creates a **support expectation mismatch** - customers buy DGX Spark for cutting-edge LLM support, but encounter framework limitations not clearly documented upfront.

## Conclusion

MiniMax-M2.5 is **architecturally incompatible** with TensorRT-LLM 1.0.0rc3. This is not a deployment, networking, or sync issue - it's a framework limitation. The model successfully synced to both DGX1 and DGX2, but TRT-LLM cannot instantiate it.

This reflects a broader issue: TensorRT-LLM's architecture support is fragmented, and DGX Spark users may encounter similar limitations with other MoE or newer models.

**Recommendation:** Use Qwen2.5-Coder-7B for production multi-node inference on DGX Spark cluster. For future purchases, validate model+TRT-LLM version compatibility with NVIDIA support before committing to hardware.

---
**Date Tested:** 2026-03-17
**Environment:** DGX1 + DGX2 (SM121 Blackwell, 200Gbps RoCE)
**TRT-LLM Version:** 1.0.0rc3 (nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3)

## References

- [NVIDIA TensorRT-LLM Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [NVIDIA TensorRT-LLM Documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [GitHub Issue #8474: Can't run GPT-OSS models on DGX Spark](https://github.com/NVIDIA/TensorRT-LLM/issues/8474)
- [GitHub Issue #11368: FP4 CUTLASS GEMM fails on GB10 (SM121)](https://github.com/NVIDIA/TensorRT-LLM/issues/11368)
- [GitHub Issue #6193: Can't load gemma3 models](https://github.com/NVIDIA/TensorRT-LLM/issues/6193)
- [GitHub Issue #5594: Will there be support for the architecture 101?](https://github.com/NVIDIA/TensorRT-LLM/issues/5594)
- [DGX Spark - TRT LLM for Inference (NVIDIA Build)](https://build.nvidia.com/spark/trt-llm)
- [vLLM Issue #36821: No sm_121 (Blackwell) support on aarch64](https://github.com/vllm-project/vllm/issues/36821)
