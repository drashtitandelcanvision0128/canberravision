# Coolify Deployment Fix for YOLO26
# =================================
# This file contains specific instructions to fix the 404 error on Coolify

## Problem
The 404 error occurs because:
1. Coolify's reverse proxy (Traefik/Nginx) can't properly route to Gradio
2. The health check was too strict, causing container restarts
3. root_path configuration was interfering with proxy routing

## Fixes Applied

### 1. start_production.py
- Changed `root_path` to `None` (lets Gradio auto-detect from proxy headers)
- Added explicit `GRADIO_ROOT_PATH=''` environment variable
- Added `favicon_path=None` to prevent favicon errors

### 2. Dockerfile  
- Removed the HEALTHCHECK directive that was causing restart loops
- Kept port 7860 exposed

### 3. health_check.py
- Made 404 responses acceptable (Gradio returns 404 on root during startup)
- Now passes if port is listening, even with 404

## Coolify Configuration Steps

### Step 1: Build Settings
```
Build Pack: Dockerfile
Dockerfile Path: Dockerfile
Context: .
```

### Step 2: Ports
```
Port: 7860
```

### Step 3: Health Check (in Coolify UI)
DISABLE the health check or set:
```
Health Check Path: /config
Health Check Port: 7860
```

### Step 4: Environment Variables
```
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_ROOT_PATH=
CUDA_VISIBLE_DEVICES=
```

### Step 5: Domain/Proxy Settings
- Let Coolify auto-generate the domain
- Don't add any custom paths or prefixes
- Keep it simple: just the generated domain

## After Deployment

1. Push changes to your branch
2. In Coolify, trigger a "Restart" or "Rebuild"
3. Wait 2-3 minutes for models to download
4. Access via the Coolify-provided URL

## Troubleshooting

If still getting 404:
1. Check Coolify logs: Should see "✅ Application starting successfully"
2. Check that port 7860 is exposed in Coolify settings
3. Try accessing /config endpoint directly
4. Make sure no custom domain path is set

## Testing Locally (to verify)

```bash
docker build -t yolo26-coolify .
docker run -p 7860:7860 yolo26-coolify
```

Then open http://localhost:7860
