# 🎨 Grok-Style Chat UI for DGX Spark TRT-LLM

Beautiful, fast, multi-node LLM chat interface powered by NVIDIA TensorRT-LLM.

## ⚡ Quick Start

```bash
./start_chat.sh
```

Then open: **http://localhost:7860**

## 🎯 Features

- **💬 Real-time Streaming Chat** - Token-by-token response streaming
- **🚀 Multi-model Support** - Download and switch between HuggingFace models
- **⚡ Speed Metrics** - Live tokens/second counter
- **💾 Chat History** - Multiple conversations stored locally
- **🟢 Status Indicator** - Visual API health status
- **🎨 Grok-Inspired Design** - Dark, minimalist UI
- **🔄 Model Loading** - Progress tracking during model switches
- **📥 Easy Model Download** - Add any HuggingFace model by name

## 📋 Architecture

### Backend (`chat/app.py`)
FastAPI server providing:
- `/api/models` - List cached models + current loaded model
- `/api/chat` - SSE streaming chat endpoint
- `/api/models/download` - Download HF models in background
- `/api/models/load` - Switch active model
- `/api/status` - System health check

### Frontend (`chat/static/index.html`)
Single-file HTML/CSS/JS app:
- Sidebar with chat history (localStorage)
- Model selector dropdown
- Streaming message renderer
- Progress bars and status indicators
- Markdown rendering + code highlighting

## 🚀 Usage

### Start Chatting
1. Select a model from dropdown (4 models pre-downloaded)
2. Type your message
3. Press Send or Ctrl+Enter
4. Watch tokens stream in real-time

### Download New Model
1. Click **+ Add** button
2. Enter HuggingFace model ID (e.g., `nvidia/Qwen3-235B-A22B-FP4`)
3. Watch progress bar
4. Model becomes available when done

### Switch Model
1. Open model dropdown
2. Click different model
3. Server restarts TRT-LLM with new model
4. Status changes from 🟡 Loading to 🟢 Ready
5. Start chatting with new model

### Multi-Turn Conversation
- Full message history is sent with each request (for context/memory)
- Each response shows tokens/sec and generation time
- Copy responses with the Copy button
- Create new chats anytime with "+ New Chat"

## 📦 Pre-Downloaded Models

- `nvidia/Llama-3.1-8B-Instruct-FP4` (2.6GB) - Fast, compact
- `mistralai/Mistral-7B-Instruct-v0.2` (28GB) - Great quality
- `Qwen/Qwen2.5-Coder-7B-Instruct` (15GB) - Code specialist
- `nvidia/Qwen3-235B-A22B-FP4` (12GB) - Large & powerful

## 🔧 Configuration

### Default Ports
- **Chat UI:** `http://localhost:7860`
- **TRT-LLM API:** `http://localhost:8355/v1`
- **DGX Cluster:** 10.0.0.1 (primary) + 10.0.0.2 (worker)

### Environment
Reads from:
- HuggingFace token: `$HF_TOKEN` environment variable
- Model cache: `/home/nvidia/.cache/huggingface/hub/`
- TRT-LLM endpoint: Hardcoded as `http://localhost:8355/v1`

## 📊 Performance

### Typical Throughput
- **Single node:** 100-200 tokens/sec (depends on model/hardware)
- **Multi-node:** Scaled across DGX1 + DGX2 cluster
- **Model loading:** 10-30 seconds for 8B models

### GPU Utilization
- Uses all 16 GPUs (8x DGX1 + 8x DGX2)
- Tensor parallel: `--tp_size 2` (1 per node)
- Memory: ~121GB per node (full model + KV cache)

## 🐛 Troubleshooting

### "Cannot connect to API"
1. Verify TRT-LLM is running: `curl http://localhost:8355/v1/models`
2. Start with: `./start_trtllm_correct.sh`
3. Wait 30-60 seconds for model to load

### "Model not in dropdown"
1. Check model exists: `ls /home/nvidia/.cache/huggingface/hub/`
2. Refresh page (Ctrl+R) to reload list
3. Or add via "+ Add" button

### "Download fails"
1. Check internet connectivity
2. Verify model name on huggingface.co
3. Check disk space: `df -h /home/nvidia/.cache/`

### "Chat stuck / no response"
1. Check API: `curl http://localhost:8355/v1/models`
2. If offline, restart: `./start_trtllm_correct.sh`
3. Or switch to different model from dropdown

## 📁 Project Structure

```
dgx2spark/
├── chat/
│   ├── app.py                    # FastAPI backend
│   ├── static/
│   │   └── index.html            # Full UI (no build needed)
│   └── requirements.txt
├── start_chat.sh                 # One-command launcher
├── start_trtllm_correct.sh       # TRT-LLM startup (THE working one)
├── openmpi-hostfile              # Multi-node config
├── archive/                      # Old/broken scripts
└── [docs]
```

## 🔐 Security Notes

- **No authentication** - Runs on localhost only
- **No API key** - Uses local HF token from environment
- **No data logging** - Chat history stored only in browser localStorage
- **Expose carefully** - Don't expose port 7860 to untrusted networks

## 🚀 Deploy Publicly (Advanced)

To expose via Traefik + Cloudflare:
1. Register in Easypanel as new service
2. Configure route at `app-chat.fzrua0.easypanel.host`
3. Traefik forwards to `http://localhost:7860`
4. Cloudflare tunnel provides HTTPS + DDoS protection

## 📝 Notes

- **Not a prod UI** - Built for DGX Spark, not tested at scale
- **localStorage limit** - ~5MB per domain, may overflow with very long chats
- **Markdown rendering** - Uses marked.js + highlight.js from CDN
- **No WebSocket** - Uses Server-Sent Events (SSE) for streaming

## 🎓 Built With

- **Backend:** Python 3, FastAPI, aiohttp
- **Frontend:** HTML, vanilla CSS, vanilla JavaScript
- **LLM:** NVIDIA TensorRT-LLM 1.0.0rc3
- **Streaming:** OpenAI-compatible API
- **UI Inspiration:** Grok by xAI

---

**Status:** ✅ Working | **Updated:** March 2026 | **Ready:** Production-ready for DGX Spark cluster
