#!/bin/bash
# Runs shuki against this directory with a prompt to upgrade prints to logging.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORKSPACE_ROOT="$SCRIPT_DIR" python3 "$SCRIPT_DIR/../main.py" \
  "let's replace prints with a more sophisticated logging system, verbose messages should be 'debug' type, other messages should be 'info' or 'error'"
