from pathlib import Path
import os
import sys

# Add the parent directory to the Python path so we can import backend module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app

# Export the app for Vercel
__all__ = ['app']
