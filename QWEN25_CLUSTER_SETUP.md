# TRT-LLM Multi-Node Setup for Qwen2.5-Coder

Following NVIDIA's official guide with fixes for GB10 issues.

**Key Differences from NVIDIA Guide:**
- Image: `1.0.0rc3` (NOT rc6 which is broken for GB10)
- Entrypoint: `bash` (NOT `sh` - shell compatibility issue)
- Model: `Qwen2.5-Coder-7B-Instruct` (smaller test model)
- Port: 8005 (custom, avoid conflicts)

---

## STEP 1: Network Setup ✅ (Already Done)
- [x] QSFP cable connected
- [x] IP addresses: 10.0.0.1 and 10.0.0.2
- [x] Ping works: `ping 10.0.0.2`

---

## STEP 2: Docker Permissions ✅
Check if docker is accessible:

```bash
docker ps
```

If permission denied, run:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## STEP 3: Create OpenMPI Hostfile

Get IP addresses from both nodes:

```bash
# On DGX1:
ip a show enp1s0f0np0 | grep inet

# On DGX2 (SSH):
ssh root@10.0.0.2 "ip a show enp1s0f0np0 | grep inet"
```

Create hostfile on DGX1:
```bash
cat > ~/openmpi-hostfile <<EOF
10.0.0.1
10.0.0.2
EOF

cat ~/openmpi-hostfile
```

Expected output:
```
10.0.0.1
10.0.0.2
```

---

## STEP 4: Start Containers on Both Nodes

**CRITICAL FIX:** Use `bash` not `sh`, and use `1.0.0rc3` not rc6

### On DGX1:
```bash
docker run -d --rm \
  --name trtllm-multinode \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
  bash -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | bash"
```

**Expected output:**
```
abc123def456
```
(Container ID)

### On DGX2 (via SSH):
```bash
ssh root@10.0.0.2 << 'SSH_CMD'
docker run -d --rm \
  --name trtllm-multinode \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3 \
  bash -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | bash"
SSH_CMD
```

---

## STEP 5: Verify Containers Are Running

On DGX1:
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

Expected:
```
NAMES               STATUS
trtllm-multinode    Up X seconds
```

On DGX2:
```bash
ssh root@10.0.0.2 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

---

## STEP 6: Copy Hostfile to Container

On DGX1:
```bash
docker cp ~/openmpi-hostfile trtllm-multinode:/etc/openmpi-hostfile
```

Verify:
```bash
docker exec trtllm-multinode cat /etc/openmpi-hostfile
```

Expected:
```
10.0.0.1
10.0.0.2
```

---

## STEP 7: Save Container Reference

```bash
export TRTLLM_MN_CONTAINER=trtllm-multinode
echo $TRTLLM_MN_CONTAINER
```

---

## STEP 8: Generate Configuration File

```bash
docker exec $TRTLLM_MN_CONTAINER bash -c 'cat <<EOF > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
EOF'
```

Verify:
```bash
docker exec $TRTLLM_MN_CONTAINER cat /tmp/extra-llm-api-config.yml
```

---

## STEP 9: Download Model

Set HF token (if not already set):
```bash
export HF_TOKEN="your-huggingface-token-here"
```

Download Qwen2.5-Coder:
```bash
docker exec \
  -e MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" \
  -e HF_TOKEN=$HF_TOKEN \
  $TRTLLM_MN_CONTAINER bash -c 'mpirun -x HF_TOKEN bash -c "hf download Qwen/Qwen2.5-Coder-7B-Instruct"'
```

**Expected output:**
- Model download progress (~14GB for 7B model)
- May take 5-10 minutes depending on network

---

## STEP 10: Serve the Model

```bash
docker exec \
  -e MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" \
  -e HF_TOKEN=$HF_TOKEN \
  $TRTLLM_MN_CONTAINER bash -c '
    mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve Qwen/Qwen2.5-Coder-7B-Instruct \
      --tp_size 2 \
      --backend pytorch \
      --max_num_tokens 8192 \
      --max_batch_size 2 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml \
      --port 8355'
```

**Expected output:**
```
[TensorRT-LLM] TensorRT LLM version: 1.0.0
...
mpi_rank: 0
mpi_rank: 1
tllm_mpi_size: 2
...
[INFO] Uvicorn running on http://0.0.0.0:8355
```

This will take 2-3 minutes for first run (engine compilation).

---

## STEP 11: Validate API Server

Open a NEW terminal (keep server running in old terminal):

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64
  }' | jq '.choices[0].message.content'
```

**Expected output:**
```json
"The answer to 2 + 2 is 4."
```

---

## STEP 12: Cleanup

Stop containers on DGX1:
```bash
docker stop trtllm-multinode
```

Stop container on DGX2:
```bash
ssh root@10.0.0.2 "docker stop trtllm-multinode"
```

Remove models (optional):
```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2*
```

---

## Troubleshooting

### Issue: "sh: curl: command not found"
**Cause:** Container doesn't have curl (or using wrong shell)
**Fix:** Already fixed by using `bash` instead of `sh`

### Issue: "permission denied while trying to connect to Docker daemon"
**Cause:** User not in docker group
**Fix:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "mpirun: command not found"
**Cause:** OpenMPI not installed in container
**Fix:** The entrypoint script installs it. Check logs:
```bash
docker logs trtllm-multinode | tail -50
```

### Issue: "NCCL timeout"
**Cause:** Inter-node communication overloaded
**Fix:** Reduce batch size or try hybrid TP+PP instead

### Issue: "ssh: permission denied"
**Cause:** SSH keys not set up
**Fix:** On DGX1, generate keys:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@10.0.0.2
```

---

## Success Criteria

- [x] Both containers running
- [x] Hostfile copied to primary container
- [x] Model downloaded
- [x] Server startup completes
- [x] API responds to test request
- [x] Response is valid JSON with generated text

---

## Notes

- **First run takes longer:** Engine compilation (~2-3 min for Qwen2.5-Coder)
- **Subsequent runs:** Much faster (engine cached)
- **GPU memory:** ~14GB per node for 7B model
- **Inter-node traffic:** TP=2 means model split across nodes (need QSFP)
