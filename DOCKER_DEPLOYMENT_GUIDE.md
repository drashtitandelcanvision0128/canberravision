# 🐳 Docker Deployment Guide for Coolify

## 📋 Overview

This guide provides complete instructions for deploying the YOLO Car Plate Detection System on Coolify using Docker containers.

## 🚀 Quick Deployment Options

### Option 1: Using Docker Compose (Recommended for Local Development)

```bash
# Clone your repository
git clone <your-repo-url>
cd YOLO26

# Copy environment file
cp .env.docker .env

# Build and start all services
docker-compose up -d

# Access services:
# Main App: http://localhost:7860
# Plate Detector: http://localhost:7861
# Enhanced Detector: http://localhost:7862
# Guaranteed Detector: http://localhost:7863
# Simple Detector: http://localhost:7864
# Parking Dashboard: http://localhost:5000
```

### Option 2: Using Coolify (Production Deployment)

#### Step 1: Prepare Your Repository
1. Push all Docker files to your Git repository
2. Ensure your repository contains:
   - `Dockerfile`
   - `docker-compose.yml`
   - `.dockerignore`
   - `coolify-deployment.yml`
   - `.env.docker`

#### Step 2: Add Application in Coolify
1. Login to your Coolify dashboard
2. Click **"Add Application"**
3. Select **"Git"** as source
4. Connect your Git repository
5. Choose your branch (usually `main` or `master`)

#### Step 3: Configure Application
1. **Application Name**: `YOLO Car Detection`
2. **Build Type**: Docker Compose
3. **Docker Compose File**: `docker-compose.yml`
4. **Environment File**: `.env.docker`

#### Step 4: Deploy
1. Click **"Deploy"**
2. Wait for the build to complete
3. Access your applications via the provided URLs

## 🔧 Configuration Files

### Dockerfile
- **Base Image**: Python 3.10-slim
- **GPU Support**: CUDA enabled
- **Dependencies**: All required packages from `requirements.txt`
- **Exposed Ports**: 7860-7864, 5000

### Docker Compose Services
| Service | Port | Description | Command |
|---------|------|-------------|---------|
| main-app | 7860 | Main YOLO application | `python apps/app.py` |
| plate-detector | 7861 | License plate image detector | `python apps/license_plate_image_detector.py` |
| enhanced-detector | 7862 | Enhanced angle detector | `python apps/enhanced_plate_detector.py` |
| guaranteed-detector | 7863 | Guaranteed plate detector | `python apps/guaranteed_plate_detector.py` |
| simple-detector | 7864 | Simple working detector | `python apps/simple_working_plate_detector.py` |
| parking-dashboard | 5000 | Parking management dashboard | `python apps/parking_dashboard.py` |
| nginx | 80/443 | Reverse proxy (optional) | - |

## 🌐 Access URLs

### Direct Port Access
- **Main Application**: `http://your-domain:7860`
- **Plate Detector**: `http://your-domain:7861`
- **Enhanced Detector**: `http://your-domain:7862`
- **Guaranteed Detector**: `http://your-domain:7863`
- **Simple Detector**: `http://your-domain:7864`
- **Parking Dashboard**: `http://your-domain:5000`

### With Nginx Reverse Proxy
- **Main Application**: `http://your-domain/`
- **Plate Detector**: `http://your-domain/plate-detector/`
- **Enhanced Detector**: `http://your-domain/enhanced/`
- **Guaranteed Detector**: `http://your-domain/guaranteed/`
- **Simple Detector**: `http://your-domain/simple/`
- **Parking Dashboard**: `http://your-domain/parking/`

## 🔒 Environment Variables

Key environment variables in `.env.docker`:

```bash
# Application Ports
GRADIO_SERVER_PORT=7860
PLATE_DETECTOR_PORT=7861
ENHANCED_DETECTOR_PORT=7862
GUARANTEED_DETECTOR_PORT=7863
SIMPLE_DETECTOR_PORT=7864
PARKING_DASHBOARD_PORT=5000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.5;8.0;8.6

# OCR Configuration
PADDLEOCR_USE_GPU=True
PADDLEOCR_GPU_MEM=500
```

## 📊 Resource Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 10GB
- **GPU**: Optional (recommended for better performance)

### Recommended Requirements
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 20GB
- **GPU**: NVIDIA GPU with CUDA support

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Available
```bash
# Check if CUDA is available in container
docker exec -it <container-name> nvidia-smi

# If not available, run without GPU
docker-compose up -d --force-recreate
```

#### 2. Port Conflicts
```bash
# Check which ports are in use
netstat -tulpn | grep :7860

# Change ports in docker-compose.yml if needed
```

#### 3. Memory Issues
```bash
# Check container resource usage
docker stats

# Increase memory limits in Coolify dashboard
```

#### 4. Build Failures
```bash
# Rebuild without cache
docker-compose build --no-cache

# Check build logs
docker-compose logs <service-name>
```

### Health Checks

Add health checks to your `docker-compose.yml`:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7860"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 🚀 Scaling

### Horizontal Scaling
In Coolify, you can scale individual services:

1. Go to your application
2. Click on the service you want to scale
3. Update **Replicas** count
4. Redeploy

### Load Balancing
The nginx reverse proxy automatically distributes load across replicas.

## 📝 Monitoring

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs main-app

# Follow logs in real-time
docker-compose logs -f
```

### Metrics
- Use Coolify's built-in monitoring
- Consider adding Prometheus/Grafana for advanced metrics

## 🔄 Updates

### Updating Your Application
1. Push changes to your Git repository
2. In Coolify, click **"Redeploy"**
3. Coolify will automatically pull changes and rebuild

### Zero-Downtime Updates
- Use rolling updates in Coolify
- Set health checks for smooth transitions

## 🛡️ Security

### Best Practices
1. Use HTTPS in production
2. Set up authentication
3. Restrict access to sensitive ports
4. Regularly update base images
5. Use secrets for sensitive data

### SSL/HTTPS Setup
```bash
# Generate SSL certificates
certbot certonly --webroot -w /var/www/html -d your-domain.com

# Update nginx.conf to use SSL
# See SSL configuration section in nginx.conf
```

## 📞 Support

If you encounter issues:

1. Check Coolify logs
2. Review container logs
3. Verify environment variables
4. Test locally with `docker-compose`
5. Check resource allocation

---

**🎉 Your YOLO Car Plate Detection System is now ready for production deployment on Coolify!**
