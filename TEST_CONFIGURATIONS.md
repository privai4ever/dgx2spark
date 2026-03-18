# DGX Spark TRT-LLM Test Configurations Log

## Test #1: Qwen2.5-Coder-7B (1.0.0rc3)
**Status:** ✅ WORKING

- TRT-LLM: nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: Qwen/Qwen2.5-Coder-7B-Instruct (15GB)
- TP Size: 2 (multi-node)
- Load Time: 15-20 min
- API: ✅ Ready
- Tokens/sec: 120-150

**Result:** Multi-node working perfectly with smaller model.

---

## Test #2: MiniMax-M2.5 (1.0.0rc3)
**Status:** ❌ FAILED & REMOVED

- TRT-LLM: nccr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: MiniMaxAI/MiniMax-M2.5 (198GB)
- Error: ValueError - Unknown architecture MiniMaxM2ForCausalLM
- Action: Deleted from cache (freed 198GB)

**Result:** Framework incompatibility, not network issue.

---

## Test #3: Qwen3-235B (1.0.0rc3)
**Status:** ⏳ SLOW - Bottleneck

- TRT-LLM: nccr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: nvidia/Qwen3-235B-A22B-FP4 (12GB FP4)
- TP Size: 2
- Issue: Flashinfer JIT compilation 15-25 min (UNDOCUMENTED)
- DGX2: Container failed initial start

**Finding:** GB10 ARM64 kernel compilation is main bottleneck.

---

## Test #4: Qwen3-235B (1.0.0rc3 FINAL) ⭐ WORKING
**Status:** ✅ FULLY OPERATIONAL

- TRT-LLM: nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 (REVERTED - stable version)
- Model: nvidia/Qwen3-235B-A22B-FP4 (12GB FP4)
- TP Size: 2 (multi-node: rank 0 on DGX1, rank 1 on DGX2)
- Containers: DGX1 ✅ DGX2 ✅
- MPI: 2 workers initialized ✅
- NCCL: Active ✅
- Load Time: ~1 minute (faster than expected - possible prebuilt kernels)
- API: ✅ Ready at localhost:8355/v1/models
- Chat UI: ✅ Ready at localhost:7860
- Tokens/sec: Streaming confirmed

### Key Success Factors
✅ Reverted to proven 1.0.0rc3 image (not latest which had NVML issues)
✅ Used correct environment variables for InfiniBand (enp1s0f0np0, enp1s0f1np1)
✅ Hostfile properly copied to both containers
✅ MPI configuration with -npernode 1 for proper rank distribution
✅ NCCL debug output shows both ranks communicating

### Results
- Model fully loaded and responding
- Streaming responses confirmed with curl
- Chat interface successfully communicating
- Multi-node inference functioning correctly

---

## Summary

Test #1: Qwen2.5 (7B) → ✅ WORKS (15-20 min, single-node)
Test #2: MiniMax (198GB) → ❌ INCOMPATIBLE (architecture unsupported)
Test #3: Qwen3-235B (12GB) → ⏳ SLOW (30-40 min, initial 1.0.0rc3 attempt)
Test #4: Qwen3-235B (12GB) → ❌ FAILED (latest image, NVML errors)
Test #5: Qwen3-235B (12GB) → ✅ WORKING (1.0.0rc3 reverted, multi-node ready)

**KEY FINDINGS:**
1. 1.0.0rc3 is the proven stable version for SM121 (not "latest")
2. Latest TRT-LLM image has NVML initialization issues on ARM64 GB10
3. Qwen3-235B loads much faster on second attempts (~1 min vs 15-25 min)
4. Multi-node setup with proper MPI/NCCL config works reliably
5. FP4 quantization is efficient - only 12GB for 235B parameter model

---

Updated: 2026-03-18 23:50 (Qwen3-235B multi-node confirmed working)
