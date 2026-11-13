#!/bin/sh
set -eu
mkdir -p "$HOME" data models results .cache
if [ "${1:-}" = "streamlit" ]; then
    python scripts/download_model.py
    python scripts/download_tracknet_model.py
fi
exec "$@"
