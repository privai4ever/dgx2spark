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

## Conclusion

MiniMax-M2.5 is **architecturally incompatible** with TensorRT-LLM 1.0.0rc3. This is not a deployment, networking, or sync issue - it's a framework limitation. The model successfully synced to both DGX1 and DGX2, but TRT-LLM cannot instantiate it.

**Recommendation:** Use Qwen2.5-Coder-7B for production multi-node inference on DGX Spark cluster.

---
**Date Tested:** 2026-03-17
**Environment:** DGX1 + DGX2 (SM121 Blackwell, 200Gbps RoCE)
**TRT-LLM Version:** 1.0.0rc3 (nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3)
