#!/usr/bin/env bash
# One-time setup for macOS
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3 first: brew install python"; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p face_data
echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python3 face_app.py collect"
echo "  python3 face_app.py recognize"
