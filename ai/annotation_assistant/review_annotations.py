#!/usr/bin/env python3
"""
Launcher script for Heritage AI Monument Decay Annotation Reviewer.
Starts local web interface on http://127.0.0.1:5500
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from review_server import start_server

if __name__ == "__main__":
    start_server(host="127.0.0.1", port=5500)
