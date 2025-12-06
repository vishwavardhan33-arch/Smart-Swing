import json
import os
from pathlib import Path
from datetime import datetime

class DataManager:
    """Manages session storage and data persistence"""
    
    def __init__(self, data_dir="swing_sessions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.sessions_file = self.data_dir / "sessions.json"
    
    def save_session(self, session_data):
        """Save session data to file"""
        sessions = self.load_all_sessions()
        sessions.append(session_data)
        
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2, default=str)
        
        print(f"[DATA_MANAGER] Session saved: {session_data['session_name']}")
    
    def load_all_sessions(self):
        """Load all sessions from file"""
        if not self.sessions_file.exists():
            return []
        
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def get_session_by_name(self, name):
        """Retrieve specific session by name"""
        sessions = self.load_all_sessions()
        for session in sessions:
            if session.get('session_name') == name:
                return session
        return None
    
    def get_statistics(self):
        """Calculate aggregate statistics across all sessions"""
        sessions = self.load_all_sessions()
        
        if not sessions:
            return None
        
        scores = [s['score'] for s in sessions]
        
        stats = {
            'total_sessions': len(sessions),
            'average_score': sum(scores) / len(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_trend': 'Improving' if scores[-1] > scores else 'Declining'
        }
        
        return stats
