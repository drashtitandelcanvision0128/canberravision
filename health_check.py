#!/usr/bin/env python3
"""
Health check script for Coolify deployment
This script verifies that the Gradio application is running and accessible.
"""

import requests
import sys
import time
import os

def health_check():
    """Perform health check on the application"""
    
    # Get configuration from environment
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = os.environ.get("GRADIO_SERVER_PORT", "7860")
    
    # Try different URLs for health check
    urls_to_try = [
        f"http://127.0.0.1:{server_port}/",
        f"http://{server_name}:{server_port}/",
        f"http://localhost:{server_port}/",
    ]
    
    print(f"[HEALTH_CHECK] Starting health check for port {server_port}")
    
    for url in urls_to_try:
        try:
            print(f"[HEALTH_CHECK] Trying: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"[HEALTH_CHECK] SUCCESS: Application is healthy at {url}")
                print(f"[HEALTH_CHECK] Status code: {response.status_code}")
                return True
            else:
                print(f"[HEALTH_CHECK] WARNING: Got status code {response.status_code} from {url}")
                
        except requests.exceptions.ConnectionError:
            print(f"[HEALTH_CHECK] Connection failed to {url}")
        except requests.exceptions.Timeout:
            print(f"[HEALTH_CHECK] Timeout connecting to {url}")
        except Exception as e:
            print(f"[HEALTH_CHECK] Error connecting to {url}: {e}")
    
    print(f"[HEALTH_CHECK] FAILED: Could not connect to application on any URL")
    return False

if __name__ == "__main__":
    # Wait a bit for the application to start
    print("[HEALTH_CHECK] Waiting 5 seconds for application to start...")
    time.sleep(5)
    
    success = health_check()
    sys.exit(0 if success else 1)
