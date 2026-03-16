# NVIDIA TRT-LLM Multi-Node Configuration Bug

## Problem

NVIDIAs officiella TRT-LLM multi-node setup-skript använder `sh` istället för `bash`, vilket orsakar problem på Ubuntu Linux eftersom:

1. Ubuntu använder `dash` som standard för `sh` (inte `bash`)
2. `dash` saknar vissa bash-funktioner som NVIDIAs skript förväntar sig
3. Detta leder till att skriptet inte känner igen kommandon eller miljövariabler

## Exempel från skripten

### Fel (nuvarande)
```bash
sh -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | sh"
```

### Korrekt
```bash
bash -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | bash"
```

## Påverkade filer

1. `start_trtllm_nvidia_guide.sh` - rad 24
2. `start_trtllm_multinode_NVIDIA_OFFICIAL.sh` - rad 102

## Lösning

Byt `sh` till `bash` i båda fallen för att säkerställa kompatibilitet med Ubuntu/Linux-system.

## Varför detta påverkar containrarna

När skriptet inte kan köras korrekt:
- Entrypoint-skriptet laddas inte ner korrekt
- Miljövariabler sätts inte upp
- NCCL/RDMA-konfigurationen misslyckas
- Resultatet: Containrarna startar men kan inte kommunicera med varandra

## Snabbfix

Redigera de påverkade filerna och byt `sh` mot `bash` på de nämnda raderna. Efter detta bör multi-node-klustret fungera som förväntat.

---
*Dokumentation skapad för att spara tid när andra stöter på samma problem med NVIDIAs officiella setup.*