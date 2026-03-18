# Docker Setup for DGX Spark Chat

Run the Grok-style chat UI in Docker, compatible with easypanel deployment.

## Quick Start

### Build & Run with Docker Compose

```bash
docker compose up -d
```

Chat will be available at: **http://localhost:7860**

### Manual Docker Run

```bash
docker build -t dgx-spark-chat:latest ./chat
docker run -d \
  --name dgx-spark-chat \
  --network host \
  -e TRT_LLM_API=http://localhost:8355/v1 \
  dgx-spark-chat:latest
```

## Configuration

### Environment Variables

- `TRT_LLM_API` - TensorRT-LLM API endpoint (default: `http://localhost:8355/v1`)
- `PYTHONUNBUFFERED` - Set to `1` for instant log output

### Network Mode

- **host** - Allows access to TRT-LLM API on localhost:8355
- Can be changed to bridge if API is on different host

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 2G
    reservations:
      cpus: '2'
      memory: 1G
```

## Integration with Easypanel

### Option 1: Import docker-compose.yml

1. Open easypanel on DGX1
2. Go to Compose section
3. Click "Create from code"
4. Paste contents of `docker-compose.yml`
5. Click Deploy

### Option 2: Add from Registry

1. Easypanel > Services > Add Service
2. Select "Docker Image"
3. Image: `dgx-spark-chat:latest`
4. Port: 7860
5. Network: host
6. Environment: Add `TRT_LLM_API=http://localhost:8355/v1`
7. Deploy

### Option 3: Manual easypanel Integration

```bash
# In easypanel container or host with Docker daemon access
docker run -d \
  --name dgx-spark-chat \
  --network host \
  -e TRT_LLM_API=http://localhost:8355/v1 \
  -e PYTHONUNBUFFERED=1 \
  --restart unless-stopped \
  dgx-spark-chat:latest
```

## Health Check

Container includes health check via `/api/models` endpoint:
- Interval: 30s
- Timeout: 10s
- Retries: 3
- Start period: 5s

To manually check:
```bash
curl http://localhost:7860/api/models
```

## Logs

View container logs:
```bash
docker logs dgx-spark-chat -f
```

## Troubleshooting

### "Cannot connect to TRT-LLM API"

1. Verify TRT-LLM is running: `curl http://localhost:8355/v1/models`
2. Check container network mode: Should be `host`
3. Verify API endpoint in container: `docker exec dgx-spark-chat curl http://localhost:8355/v1/models`

### Port Already in Use

```bash
# Find process using port 7860
lsof -i :7860

# Kill it
kill -9 <PID>

# Or use different port
docker run -p 8000:7860 dgx-spark-chat:latest
```

### Container Won't Start

1. Check logs: `docker logs dgx-spark-chat`
2. Verify image exists: `docker images | grep dgx-spark-chat`
3. Rebuild: `docker build -t dgx-spark-chat:latest ./chat`

## Performance Tips

1. **Use host network mode** - Lower latency for TRT-LLM API calls
2. **Pin CPU cores** - Optional but improves consistency
3. **Monitor resources** - Set appropriate memory limits
4. **Enable autorestart** - `restart: unless-stopped`

## Deployment Checklist

- [x] TRT-LLM API running on localhost:8355
- [x] Docker image built: `dgx-spark-chat:latest`
- [x] docker-compose.yml configured
- [x] Network mode set to `host`
- [x] Port 7860 accessible
- [x] Health check passing
- [x] Logs visible

## Architecture

```
┌─────────────────────────────────────┐
│         easypanel (Traefik)         │
│         (optional proxy)             │
└────────────────┬────────────────────┘
                 │
                 ↓ :7860
┌─────────────────────────────────────┐
│    dgx-spark-chat (Docker)          │
│  - FastAPI uvicorn server           │
│  - Streaming chat API               │
│  - Thinking mode UI                 │
└────────────────┬────────────────────┘
                 │ (host network)
                 ↓ :8355
┌─────────────────────────────────────┐
│   TRT-LLM (Docker container)        │
│  - Qwen3-235B multi-node inference  │
│  - OpenAI-compatible API            │
└─────────────────────────────────────┘
```

## Production Notes

1. Consider reverse proxy (Traefik/nginx) for SSL
2. Use environment file: `docker run --env-file .env ...`
3. Enable log rotation: Already configured in docker-compose.yml
4. Monitor container health: `docker ps` shows UNHEALTHY if issues
5. Back up any persistent data if needed

## Updates

Rebuild and redeploy:
```bash
docker build -t dgx-spark-chat:latest ./chat
docker compose up -d  # Auto-pulls new image and restarts
```

---

For easypanel support, see: https://easypanel.io/docs
For Docker Compose docs: https://docs.docker.com/compose/
