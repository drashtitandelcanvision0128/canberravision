#!/usr/bin/env python3
"""
Health check script for Coolify deployment
This script verifies that the Gradio application is running and accessible.
"""

import requests
import sys
import time
import os
import socket

def check_port_listening(port):
    """Check if a port is actually listening"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except:
        return False

def health_check():
    """Perform health check on the application"""
    
    # Get configuration from environment
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    print(f"[HEALTH_CHECK] Starting health check for port {server_port}")
    
    # First check if port is actually listening
    if not check_port_listening(server_port):
        print(f"[HEALTH_CHECK] FAILED: Port {server_port} is not listening")
        return False
    
    print(f"[HEALTH_CHECK] Port {server_port} is listening")
    
    # Try different URLs for health check
    urls_to_try = [
        f"http://127.0.0.1:{server_port}/",
        f"http://localhost:{server_port}/",
        f"http://127.0.0.1:{server_port}/config",
        f"http://localhost:{server_port}/config",
    ]
    
    for url in urls_to_try:
        try:
            print(f"[HEALTH_CHECK] Trying: {url}")
            response = requests.get(url, timeout=5, allow_redirects=True)
            
            if response.status_code == 200:
                print(f"[HEALTH_CHECK] SUCCESS: Application is healthy at {url}")
                print(f"[HEALTH_CHECK] Status code: {response.status_code}")
                print(f"[HEALTH_CHECK] Response length: {len(response.content)} bytes")
                return True
            elif response.status_code == 404:
                print(f"[HEALTH_CHECK] WARNING: Got 404 from {url}")
                # Try to check if it's a Gradio 404 (app running but page not found)
                # or a real failure
                if len(response.content) > 0:
                    content_sample = response.text[:200]
                    print(f"[HEALTH_CHECK] Response content sample: {content_sample}")
                # Return False for 404 as the app should serve content on root
                print(f"[HEALTH_CHECK] 404 on root path indicates app not fully ready")
                return False
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
    print("[HEALTH_CHECK] Waiting 10 seconds for application to start...")
    time.sleep(10)
    
    success = health_check()
    if success:
        print("[HEALTH_CHECK] ✅ Health check passed")
    else:
        print("[HEALTH_CHECK] ❌ Health check failed")
    
    sys.exit(0 if success else 1)
