import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}")
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration structure."""
        return {
            "github_accounts": {},
            "current_account": None,
            "repositories": {},
            "settings": {
                "default_branch": "main",
                "auto_sync": False,
                "backup_enabled": True
            }
        }
    
    def save_config(self):
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.config_file}")
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def add_github_account(self, username: str, token: Optional[str] = None, email: Optional[str] = None):
        """Add a GitHub account."""
        self.config["github_accounts"][username] = {
            "token": token,
            "email": email,
            "added_date": str(Path().absolute())
        }
        if not self.config["current_account"]:
            self.config["current_account"] = username
        self.save_config()
        print(f"Added GitHub account: {username}")
    
    def remove_github_account(self, username: str):
        """Remove a GitHub account."""
        if username in self.config["github_accounts"]:
            del self.config["github_accounts"][username]
            if self.config["current_account"] == username:
                self.config["current_account"] = next(iter(self.config["github_accounts"]), None)
            self.save_config()
            print(f"Removed GitHub account: {username}")
        else:
            print(f"Account {username} not found")
    
    def set_current_account(self, username: str):
        """Set the current active GitHub account."""
        if username in self.config["github_accounts"]:
            self.config["current_account"] = username
            self.save_config()
            print(f"Current account set to: {username}")
        else:
            print(f"Account {username} not found")
    
    def add_repository(self, name: str, url: str, branch: str = "main", account: Optional[str] = None):
        """Add a repository configuration."""
        if not account:
            account = self.config["current_account"]
        
        self.config["repositories"][name] = {
            "url": url,
            "branch": branch,
            "account": account,
            "added_date": str(Path().absolute())
        }
        self.save_config()
        print(f"Added repository: {name} ({url})")
    
    def remove_repository(self, name: str):
        """Remove a repository configuration."""
        if name in self.config["repositories"]:
            del self.config["repositories"][name]
            self.save_config()
            print(f"Removed repository: {name}")
        else:
            print(f"Repository {name} not found")
    
    def get_current_account(self) -> Optional[str]:
        """Get current active account."""
        return self.config.get("current_account")
    
    def get_accounts(self) -> List[str]:
        """Get list of all GitHub accounts."""
        return list(self.config["github_accounts"].keys())
    
    def get_repositories(self) -> Dict:
        """Get all repositories."""
        return self.config["repositories"]
    
    def get_account_info(self, username: str) -> Optional[Dict]:
        """Get account information."""
        return self.config["github_accounts"].get(username)
    
    def list_config(self):
        """Display current configuration."""
        print("\n=== Configuration Manager ===")
        print(f"Current Account: {self.config.get('current_account', 'None')}")
        
        print("\nGitHub Accounts:")
        for username, info in self.config["github_accounts"].items():
            current = " (CURRENT)" if username == self.config["current_account"] else ""
            print(f"  - {username}{current}")
            if info.get("email"):
                print(f"    Email: {info['email']}")
        
        print("\nRepositories:")
        for name, info in self.config["repositories"].items():
            print(f"  - {name}")
            print(f"    URL: {info['url']}")
            print(f"    Branch: {info['branch']}")
            print(f"    Account: {info['account']}")
        
        print("\nSettings:")
        for key, value in self.config["settings"].items():
            print(f"  {key}: {value}")

def main():
    """Command line interface for config manager."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_manager.py <command> [args]")
        print("Commands:")
        print("  add_account <username> [token] [email]")
        print("  remove_account <username>")
        print("  set_account <username>")
        print("  add_repo <name> <url> [branch] [account]")
        print("  remove_repo <name>")
        print("  list")
        print("  help")
        return
    
    config = ConfigManager()
    command = sys.argv[1]
    
    if command == "add_account" and len(sys.argv) >= 3:
        username = sys.argv[2]
        token = sys.argv[3] if len(sys.argv) > 3 else None
        email = sys.argv[4] if len(sys.argv) > 4 else None
        config.add_github_account(username, token, email)
    
    elif command == "remove_account" and len(sys.argv) >= 3:
        config.remove_github_account(sys.argv[2])
    
    elif command == "set_account" and len(sys.argv) >= 3:
        config.set_current_account(sys.argv[2])
    
    elif command == "add_repo" and len(sys.argv) >= 4:
        name = sys.argv[2]
        url = sys.argv[3]
        branch = sys.argv[4] if len(sys.argv) > 4 else "main"
        account = sys.argv[5] if len(sys.argv) > 5 else None
        config.add_repository(name, url, branch, account)
    
    elif command == "remove_repo" and len(sys.argv) >= 3:
        config.remove_repository(sys.argv[2])
    
    elif command == "list":
        config.list_config()
    
    else:
        print("Invalid command or missing arguments")

if __name__ == "__main__":
    main()
