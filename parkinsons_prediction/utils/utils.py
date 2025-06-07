
import logging
import subprocess

def get_git_commit_id():
    """Get current git commit ID."""
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit_id
    except:
        return "unknown"
    
def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )