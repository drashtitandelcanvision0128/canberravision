#!/usr/bin/env python3
"""
GitHub Setup Script for YOLO26 Project
This script helps remove current remote and add new GitHub configuration
"""

import subprocess
import sys
from config_manager import ConfigManager

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return False
        return result.stdout.strip()
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def remove_current_remote():
    """Remove the current Hugging Face remote."""
    print("Removing current remote...")
    
    # Check current remotes
    remotes = run_command("git remote -v")
    if remotes:
        print(f"Current remotes:\n{remotes}")
        
        # Remove origin remote
        if run_command("git remote remove origin"):
            print("✓ Removed Hugging Face remote")
        else:
            print("✗ Failed to remove remote")
            return False
    
    return True

def add_github_remote(username, repo_name, branch="main"):
    """Add new GitHub remote."""
    github_url = f"https://github.com/{username}/{repo_name}.git"
    
    print(f"Adding GitHub remote: {github_url}")
    
    if run_command(f"git remote add origin {github_url}"):
        print(f"✓ Added GitHub remote for {username}/{repo_name}")
        
        # Set upstream branch
        if run_command(f"git branch --set-upstream-to=origin/{branch} {branch}"):
            print(f"✓ Set upstream branch to {branch}")
        
        return True
    else:
        print("✗ Failed to add GitHub remote")
        return False

def setup_git_config(username, email):
    """Configure Git user settings."""
    print(f"Setting up Git config for {username}")
    
    if run_command(f"git config user.name '{username}'"):
        print(f"✓ Set Git username to {username}")
    
    if email and run_command(f"git config user.email '{email}'"):
        print(f"✓ Set Git email to {email}")
    
    return True

def main():
    """Main setup function."""
    print("=== YOLO26 GitHub Setup ===\n")
    
    config = ConfigManager()
    
    print("Please provide the following information:")
    
    # Get GitHub account details
    username = input("GitHub username: ").strip()
    if not username:
        print("Username is required!")
        return
    
    email = input("GitHub email (optional): ").strip() or None
    token = input("GitHub personal access token (optional, for private repos): ").strip() or None
    
    # Get repository details
    repo_name = input("Repository name (e.g., YOLO26): ").strip() or "YOLO26"
    branch = input("Branch name (default: main): ").strip() or "main"
    
    print(f"\nSetting up for: {username}/{repo_name} (branch: {branch})")
    
    # Add to config manager
    config.add_github_account(username, token, email)
    config.add_repository(repo_name, f"https://github.com/{username}/{repo_name}.git", branch, username)
    config.set_current_account(username)
    
    # Remove current remote
    if not remove_current_remote():
        print("Failed to remove current remote. Please check manually.")
        return
    
    # Add new GitHub remote
    if not add_github_remote(username, repo_name, branch):
        print("Failed to add GitHub remote. Please check manually.")
        return
    
    # Setup Git config
    setup_git_config(username, email)
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Push your code to GitHub:")
    print(f"   git push -u origin {branch}")
    print("2. Or create a new repository on GitHub first, then push")
    print("3. Check your configuration with: python config_manager.py list")
    
    # Show current configuration
    config.list_config()

if __name__ == "__main__":
    main()
