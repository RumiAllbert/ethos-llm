"""Path fixing module for scripts.

This module adds the project root directory to the Python path,
allowing scripts to import modules from the src directory.
"""

import os
import sys
from pathlib import Path

# Get the absolute path of the project root directory
# (parent directory of the scripts directory)
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent

# Add the project root to the Python path
sys.path.insert(0, str(PROJECT_ROOT))
