#!/usr/bin/env python3
"""
Script for sending macOS notifications.
"""
import sys
import subprocess
from agent.src.notification_client import NotificationClient

def main():
    """Send a notification with the given message."""
    if len(sys.argv) < 2:
        print("Usage: notify.py <message> [title] [subtitle] [--alert]")
        return 1
        
    message = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "AI System"
    subtitle = sys.argv[3] if len(sys.argv) > 3 else ""
    style = "alert" if "--alert" in sys.argv else "banner"
    
    try:
        NotificationClient.notify(
            message=message,
            title=title,
            subtitle=subtitle,
            style=style
        )
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error sending notification: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 