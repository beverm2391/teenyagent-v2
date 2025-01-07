"""
Client for sending system notifications
"""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class NotificationClient:
    """Client for sending system notifications"""
    
    @staticmethod
    def notify(message: str, title: str = "AI System", subtitle: str = "", style: str = "banner") -> None:
        """
        Send a system notification
        
        Args:
            message: The notification message
            title: The notification title (default: "AI System")
            subtitle: Optional subtitle text
            style: Notification style - "alert" or "banner" (default: "banner")
        """
        # Escape quotes in message and title
        message = message.replace('"', '\\"')
        title = title.replace('"', '\\"')
        subtitle = subtitle.replace('"', '\\"')
        
        if style == "alert":
            # Display modal alert dialog
            script = f'''
            tell application "System Events"
                display dialog "{message}" with title "{title}" buttons {{"OK"}} default button "OK"
            end tell
            '''
        else:
            # Display banner notification
            script = f'display notification "{message}" with title "{title}" sound name "default"'
            if subtitle:
                script = script.replace('with title', f'subtitle "{subtitle}" with title')
                
        try:
            subprocess.run(["osascript", "-e", script], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to send notification: {e}")
            raise 