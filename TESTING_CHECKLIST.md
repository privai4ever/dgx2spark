# TRT-LLM Multi-Node Testing Checklist

## Pre-Flight Checks

### On DGX1 (10.0.0.1):
- [ ] `nvidia-smi` shows 8 GPUs
- [ ] `ibv_devinfo` shows IB device active (PORT_ACTIVE)
- [ ] `ip link show ib0` exists and MTU=4096
- [ ] `docker info` works (no permission errors)
- [ ] `ssh 10.0.0.2 echo ok` succeeds
- [ ] HF token set: `echo $HUGGINGFACE_TOKEN`
- [ ] HF cache exists: `ls /home/ss/.cache/huggingface`

### On DGX2 (10.0.0.2):
- [ ] Same checks via SSH: `ssh 10.0.0.2 nvidia-smi`
- [ ] IB device active: `ssh 10.0.0.2 ibv_devinfo`
- [ ] Docker running: `ssh 10.0.0.2 docker info`
- [ ] HF cache accessible at same path

## Test Execution

### 1. Stop any existing containers
```bash
./start_trtllm_multinode.sh --stop
```

### 2. Cleanup any leftovers
```bash
docker ps -a | grep trtllm | awk '{print $1}' | xargs -r docker rm -f
ssh 10.0.0.2 "docker ps -a | grep trtllm | awk '{print $1}' | xargs -r docker rm -f"
```

### 3. Start cluster
```bash
./start_trtllm_multinode.sh
```

Expected output:
```
[INFO] Starting TensorRT-LLM multi-node cluster...
[INFO] Model: Qwen/Qwen2.5-Coder-7B-Instruct
[INFO] Master: 10.0.0.1 (port 29500)
[INFO] Slave: 10.0.0.2
[INFO] API endpoint: http://10.0.0.1:8000/v1/chat/completions
[INFO] InfiniBand detected - using RDMA for 200Gbps!  <-- SHOULD SEE THIS
[INFO] Using network interface: ib0
```

### 4. Check container status
```bash
./start_trtllm_multinode.sh --status
```

Should show:
- `trtllm-node-0` (DGX1) - Up
- `trtllm-node-1` (DGX2) - Up

### 5. Check logs for NCCL/IB
```bash
docker logs -f trtllm-node-0 2>&1 | grep -E "NCCL|IB|transport|rank|world"
```

**Looking for**:
```
[TRT-LLM] [I] global_steady_clock_offset at each rank: [0.0]
[NCCL] Using IB
[NCCL] Transport: IB
[rank 0] ... [rank 1] ...
```

**NOT looking for**:
```
[NCCL] Using Socket  <-- This means TCP (10G), not IB (200G)
```

### 6. Test API endpoint
Wait ~2-5 minutes for first engine build, then:

```bash
curl http://10.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 100
  }'
```

Expected: JSON response with generated text.

### 7. Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

Should see both nodes using all 8 GPUs each (16 total).

## Success Criteria

✅ **All 16 GPUs utilized** (8 on DGX1, 8 on DGX2)
✅ **NCCL using IB** (200 Gbps), not Socket (10 Gbps)
✅ **API responds** with generated text
✅ **No errors** in logs (only INFO/WARN)
✅ **Model loads** completely (no hangs)

## Known Issues & Fixes

### Issue: "Port already in use"
**Fix**: Change PORT in script or kill process using port 8000

### Issue: Container hangs at "Loading model..."
**Possible causes**:
- HF token missing/invalid → Set `export HUGGINGFACE_TOKEN=...`
- Network issue downloading model → Check internet connectivity
- Not enough GPU memory → Reduce `tensor_parallel_size` (but we need all GPUs)

### Issue: NCCL timeout / "NCCL WARN"
**Fix**: 
- Verify IB MTU=4096 on both nodes
- Check `NCCL_IB_DISABLE=0` inside container: `docker exec trtllm-node-0 env | grep NCCL`
- Verify IB devices passed through: `docker exec trtllm-node-0 ls /dev/infiniband`

### Issue: Only DGX1 GPU usage, DGX2 idle
**Fix**: 
- Check DGX2 container is running: `ssh 10.0.0.2 docker ps | grep trtllm-node-1`
- Check DGX2 logs: `ssh 10.0.0.2 docker logs trtllm-node-1`
- Verify NCCL can connect between nodes

## Next Steps After Success

1. ✅ Document successful run (update HISTORY.md)
2. 📊 Benchmark: measure tokens/sec
3. 🎯 Test with larger model (e.g., 70B parameter)
4. ⚙️ Fine-tune NCCL parameters for max performance
5. 🚀 Try Minimax-2.5 (200GB with FP8 quantization)

---

**Status**: Ready to test ⏳
