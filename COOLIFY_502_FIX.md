# COOLIFY 502 BAD GATEWAY FIX
# ============================
# This guide fixes the 502 error when Gradio works internally but fails externally

## Problem
502 Bad Gateway = Traefik cannot connect to your container port
Your app works on http://localhost:7860 inside container, but external domain fails

## Solution Steps

### 1. UPDATE DOCKERFILE (DONE)
Added `ENV PORT=7860` - this is CRITICAL for Coolify port detection

### 2. COOLIFY SERVICE CONFIGURATION

Go to your Coolify dashboard → Services → Your App → Settings:

#### General Tab:
```
Build Pack: Dockerfile
Dockerfile Path: Dockerfile
Context: .
```

#### Ports Tab (CRITICAL):
```
Container Port: 7860
Published Port: (leave empty for auto)
```

OR if using Docker Compose:
```yaml
ports:
  - "7860:7860"
```

#### Health Check Tab:
```
Health Check Path: /
Health Check Port: 7860
```

#### Environment Variables:
```
PORT=7860
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
```

### 3. TRAEFIK LABELS (if using docker-compose in Coolify)

Add these labels to your service:
```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.yolo.rule=Host(`your-domain.com`)"
  - "traefik.http.routers.yolo.entrypoints=websecure"
  - "traefik.http.services.yolo.loadbalancer.server.port=7860"
  - "traefik.http.services.yolo.loadbalancer.server.scheme=http"
```

### 4. DEPLOYMENT CHECKLIST

Before deploying, verify:
- [ ] Dockerfile has `ENV PORT=7860`
- [ ] Dockerfile has `EXPOSE 7860`
- [ ] Coolify service shows Port: 7860
- [ ] No conflicting port mappings (remove 3000, 8000, etc.)
- [ ] Health check uses port 7860

### 5. DEBUGGING COMMANDS

After deployment, run these in Coolify terminal:

```bash
# Check if app is running
curl http://localhost:7860

# Check port binding
netstat -tlnp | grep 7860

# Check process
ps aux | grep python

# Check Traefik routing (inside Coolify network)
docker network ls
docker inspect <network_name>
```

### 6. COMMON MISTAKES TO AVOID

❌ WRONG:
- Port 3000 in Coolify settings
- Missing ENV PORT=7860
- Traefik pointing to wrong port

✅ CORRECT:
- Port 7860 everywhere
- ENV PORT=7860 in Dockerfile
- Traefik → 7860

### 7. VERIFY DEPLOYMENT

After pushing changes and redeploying:

1. Check logs show: `Running on local URL: http://0.0.0.0:7860`
2. Wait 5 minutes for models to download
3. Access your domain - should load Gradio UI

## SUMMARY

The 502 error was caused by Coolify not knowing which port to route to.
By adding `ENV PORT=7860`, Coolify auto-detects the correct port.

Push the updated Dockerfile and redeploy!
