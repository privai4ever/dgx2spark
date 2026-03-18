#!/usr/bin/env python3
"""
Grok-style Chat UI for DGX Spark TRT-LLM Cluster
FastAPI backend for streaming chat, model management, and progress tracking
"""

import os
import json
import time
import asyncio
import subprocess
import aiohttp
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection pool for reuse
http_session = None

async def get_http_session():
    """Get or create persistent HTTP session."""
    global http_session
    if http_session is None:
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        http_session = aiohttp.ClientSession(connector=connector)
    return http_session

# Serve static files (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
TRT_LLM_API = "http://localhost:8355/v1"
HF_CACHE_DIR = Path("/home/nvidia/.cache/huggingface/hub")
DGX1_IP = "10.0.0.1"
DGX2_IP = "10.0.0.2"

# Global state
download_jobs: Dict[str, dict] = {}  # {job_id: {status, progress, error}}
model_loading_state = {"status": "idle", "progress": 0, "error": None}

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def get_local_models() -> List[str]:
    """Scan HF cache for downloaded models."""
    if not HF_CACHE_DIR.exists():
        return []

    models = []
    for model_dir in HF_CACHE_DIR.glob("models--*/"):
        # Convert "models--org--name" to "org/name"
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        models.append(model_name)

    return sorted(models)

async def get_loaded_model() -> Optional[str]:
    """Query TRT-LLM API for currently loaded model."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{TRT_LLM_API}/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        return data["data"][0]["id"]
    except Exception as e:
        print(f"Error getting loaded model: {e}")
    return None

async def check_api_health() -> bool:
    """Check if TRT-LLM API is ready."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{TRT_LLM_API}/models", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                return resp.status == 200
    except:
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the chat UI."""
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def favicon():
    """Return a simple favicon to avoid 404."""
    return {"status": "ok"}

@app.get("/api/models")
async def list_models():
    """Get local cached models and currently loaded model."""
    local = get_local_models()
    loaded = await get_loaded_model()
    health = await check_api_health()

    return {
        "local_models": local,
        "loaded_model": loaded,
        "api_ready": health
    }

@app.get("/api/status")
async def status():
    """Get overall system status."""
    loaded = await get_loaded_model()
    health = await check_api_health()

    return {
        "trt_llm": "ready" if health else "offline",
        "loaded_model": loaded,
        "api_url": TRT_LLM_API,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_stream(request: Request):
    """Stream chat response from TRT-LLM with token/sec tracking."""
    try:
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "")
        max_tokens = body.get("max_tokens", 512)

        if not messages:
            raise HTTPException(status_code=400, detail="No messages")

        # Prepare payload for TRT-LLM
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": max_tokens
        }

        async def generate():
            """SSE event generator with token counting."""
            start_time = time.time()
            token_count = 0

            try:
                session = await get_http_session()
                async with session.post(
                    f"{TRT_LLM_API}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status != 200:
                        yield f'data: {{"error": "API error: {resp.status}"}}\n\n'
                        return

                    buffer = ""
                    async for chunk in resp.content.iter_chunked(4096):
                        buffer += chunk.decode('utf-8', errors='ignore')
                        lines = buffer.split('\n')
                        buffer = lines[-1]  # Keep incomplete line

                        for line in lines[:-1]:
                            if not line.strip():
                                continue

                            if line.startswith("data: "):
                                try:
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        elapsed = time.time() - start_time
                                        tps = token_count / elapsed if elapsed > 0 else 0
                                        stats = {
                                            "tokens": token_count,
                                            "time": round(elapsed, 2),
                                            "tokens_per_sec": round(tps, 1)
                                        }
                                        yield f'data: {{"stats": {json.dumps(stats)}}}\n\n'
                                        continue

                                    data = json.loads(data_str)
                                    if "choices" in data:
                                        choice = data["choices"][0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")

                                        if content:
                                            token_count += 1
                                            # Optimize JSON output - minimal escaping
                                            escaped = content.replace('\\', '\\\\').replace('"', '\\"')
                                            yield f'data: {{"token": "{escaped}"}}\n\n'
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass

                    # Process any remaining buffer
                    if buffer.startswith("data: "):
                        try:
                            data_str = buffer[6:].strip()
                            if data_str == "[DONE]":
                                elapsed = time.time() - start_time
                                tps = token_count / elapsed if elapsed > 0 else 0
                                yield f'data: {{"stats": {{"tokens": {token_count}, "time": {round(elapsed, 2)}, "tokens_per_sec": {round(tps, 1)}}}}}\n\n'
                        except:
                            pass

            except asyncio.TimeoutError:
                yield f'data: {{"error": "Request timeout"}}\n\n'
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
async def download_model(request: Request):
    """Download a HuggingFace model in background."""
    try:
        body = await request.json()
        model_id = body.get("model_id", "")

        if not model_id:
            raise HTTPException(status_code=400, detail="No model_id provided")

        job_id = str(uuid4())[:8]
        download_jobs[job_id] = {
            "status": "starting",
            "progress": 0,
            "error": None,
            "model_id": model_id
        }

        # Start download in background task
        asyncio.create_task(perform_download(job_id, model_id))

        return {"job_id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_download(job_id: str, model_id: str):
    """Execute model download with progress tracking."""
    try:
        download_jobs[job_id]["status"] = "downloading"

        # Run huggingface-cli download
        process = await asyncio.create_subprocess_exec(
            "huggingface-cli", "download", model_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)

        if process.returncode == 0:
            download_jobs[job_id]["status"] = "completed"
            download_jobs[job_id]["progress"] = 100
        else:
            download_jobs[job_id]["status"] = "error"
            download_jobs[job_id]["error"] = stderr.decode()[:200]

    except asyncio.TimeoutError:
        download_jobs[job_id]["status"] = "error"
        download_jobs[job_id]["error"] = "Download timeout"
    except Exception as e:
        download_jobs[job_id]["status"] = "error"
        download_jobs[job_id]["error"] = str(e)[:200]

@app.get("/api/models/download/{job_id}")
async def get_download_status(job_id: str):
    """Get download progress via SSE."""

    async def event_generator():
        last_status = None

        while True:
            job = download_jobs.get(job_id)

            if not job:
                yield f'data: {{"error": "Job not found"}}\n\n'
                break

            if job["status"] != last_status:
                event = {
                    "status": job["status"],
                    "progress": job["progress"],
                    "error": job["error"]
                }
                yield f'data: {json.dumps(event)}\n\n'
                last_status = job["status"]

            # Check if download finished
            if job["status"] in ["completed", "error"]:
                await asyncio.sleep(0.5)
                # Clean up old jobs after a while
                if time.time() - start_time > 60:
                    del download_jobs[job_id]
                break

            await asyncio.sleep(1)

    start_time = time.time()
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/models/load")
async def load_model(request: Request):
    """Switch to a different model (stop/restart TRT-LLM)."""
    try:
        body = await request.json()
        model_id = body.get("model_id", "")

        if not model_id:
            raise HTTPException(status_code=400, detail="No model_id")

        # Start model loading in background
        asyncio.create_task(perform_model_load(model_id))

        return {"status": "loading", "model": model_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_model_load(model_id: str):
    """Execute model loading sequence."""
    global model_loading_state

    try:
        model_loading_state = {"status": "stopping_old_model", "progress": 10, "error": None}

        # Stop containers on both nodes
        for dgx_ip in [DGX1_IP, DGX2_IP]:
            try:
                if dgx_ip == DGX1_IP:
                    subprocess.run(["docker", "stop", "trtllm-multinode"], timeout=30)
                else:
                    subprocess.run(
                        ["ssh", f"nvidia@{dgx_ip}", "docker", "stop", "trtllm-multinode"],
                        timeout=30
                    )
            except Exception as e:
                print(f"Error stopping container on {dgx_ip}: {e}")

        model_loading_state = {"status": "starting_new_model", "progress": 30, "error": None}

        # Start containers with new model
        env = os.environ.copy()
        env["MODEL"] = model_id

        subprocess.Popen(
            ["bash", "start_trtllm_correct.sh"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        model_loading_state = {"status": "waiting_for_api", "progress": 60, "error": None}

        # Poll for API readiness
        for _ in range(300):  # 5 minute timeout (large models need time)
            if await check_api_health():
                model_loading_state = {"status": "ready", "progress": 100, "error": None}
                return
            await asyncio.sleep(1)

        model_loading_state = {"status": "error", "progress": 0, "error": "API not ready after 5 minutes"}

    except Exception as e:
        model_loading_state = {"status": "error", "progress": 0, "error": str(e)[:200]}

@app.get("/api/models/loading-status")
async def get_loading_status():
    """Get current model loading status."""
    return model_loading_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
