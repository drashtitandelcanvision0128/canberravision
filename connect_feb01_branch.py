#!/usr/bin/env python3
"""
Connect to feb.0.1 branch on GitHub
"""

import subprocess
import sys
from config_manager import ConfigManager

def run_command(command):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return result.stdout.strip()
    except Exception as e:
        print(f"Exception: {e}")
        return False

def connect_to_branch(username, repo_name="YOLO26", branch="feb.0.1"):
    """Connect to specific branch on GitHub."""
    
    # Add GitHub remote
    github_url = f"https://github.com/{username}/{repo_name}.git"
    print(f"Adding remote: {github_url}")
    
    if run_command(f"git remote add origin {github_url}"):
        print("✓ Added GitHub remote")
    
    # Fetch the branch
    print(f"Fetching branch: {branch}")
    if run_command(f"git fetch origin {branch}"):
        print(f"✓ Fetched {branch}")
    
    # Create and checkout the branch locally
    print(f"Creating local branch: {branch}")
    if run_command(f"git checkout -b {branch} origin/{branch}"):
        print(f"✓ Created and checked out {branch}")
    
    # Set upstream
    if run_command(f"git branch --set-upstream-to=origin/{branch} {branch}"):
        print(f"✓ Set upstream to origin/{branch}")
    
    # Update config manager
    config = ConfigManager()
    config.add_github_account(username)
    config.add_repository(repo_name, github_url, branch, username)
    config.set_current_account(username)
    
    print(f"\n✓ Successfully connected to {username}/{repo_name} branch: {branch}")
    print("You can now push changes with: git push origin feb.0.1")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python connect_feb01_branch.py <github_username> [repo_name]")
        sys.exit(1)
    
    username = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else "YOLO26"
    
    connect_to_branch(username, repo_name)
