# Coolify Deployment Guide for YOLO Vision System

## Fixed Issues ✅

The 502 Bad Gateway error was caused by missing environment variables and timing issues in the Docker configuration.

## Changes Made:

1. **apps/app.py**: 
   - Now reads `GRADIO_SERVER_NAME` from environment variables
   - Added comprehensive debugging output
   - Added `prevent_thread_lock=False` for better container compatibility
   - Better error handling and logging

2. **docker-compose.yml**: Added missing environment variables:
   - `GRADIO_SERVER_NAME=0.0.0.0`
   - `APP_ENV=production`

3. **Dockerfile**: 
   - Updated health check with custom script
   - Added startup script for proper initialization
   - Increased start period to 180s

4. **New Files**:
   - `health_check.py`: Custom health check script
   - `start.sh`: Startup script with environment debugging

## Coolify Deployment Steps:

### Option 1: Using Dockerfile (Recommended)
1. In Coolify, create a new application
2. Set the source to your Git repository
3. Use the updated Dockerfile
4. Add these environment variables in Coolify:
   ```
   GRADIO_SERVER_PORT=7860
   GRADIO_SERVER_NAME=0.0.0.0
   APP_ENV=production
   ```

### Option 2: Using Docker Compose
1. In Coolify, select "Docker Compose" as the deployment type
2. Use the updated docker-compose.yml file
3. The environment variables are now included in the file

## Environment Variables (Complete List):
```
APP_ENV=production
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
# Add your other environment variables here...
```

## Port Configuration:
- Main app: 7860
- Plate detector: 7861
- Enhanced detector: 7862
- Guaranteed detector: 7863
- Simple detector: 7864
- Parking dashboard: 5000

## Health Check:
The custom health check script:
- Waits 5 seconds for application startup
- Tests multiple URLs for connectivity
- Provides detailed logging
- Runs every 30 seconds with 180s startup period

## Troubleshooting:
If you still get 502 errors:
1. **Check container logs** in Coolify - look for the debugging output
2. **Verify environment variables** are set correctly in Coolify UI
3. **Wait for full startup** - the application now needs up to 3 minutes
4. **Check health status** - the container should show "healthy" green status
5. **Port mapping** - ensure 7860:7860 is correctly mapped

## Deployment Process:
1. Push all changes to your Git repository
2. In Coolify, trigger a redeploy (or restart the service)
3. Wait for the container to become "healthy" (up to 3 minutes)
4. Check the logs for debugging information
5. Access your application at the Coolify domain

## Verification:
After deployment, your application should be accessible at:
- Your Coolify domain + port 7860
- Example: `https://your-app.coolify.app:7860`

The application will now properly bind to 0.0.0.0, provide detailed startup logs, and handle container health checks correctly.
