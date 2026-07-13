#!/usr/bin/env bash
# Create (or update) the conda env from environment.yml, then run the
# installation smoke tests inside it. Single entry point so you never have
# to remember to (re-)run `conda env create` before testing.
#
# Usage: ./scripts/verify_install.sh [pytest args...]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_name="$(sed -n 's/^name:[[:space:]]*//p' environment.yml | head -1)"

conda env create -f environment.yml -n "$env_name" ||
    conda env update -f environment.yml -n "$env_name" --prune

# Resolve the env's own interpreter directly rather than going through
# `conda run`/`conda activate`, whose PATH handling is unreliable when the
# invoking shell already has a different env's bin/ earlier on PATH.
env_prefix="$(conda env list | awk -v name="$env_name" '$1 == name {print $NF}')"
if [ -z "$env_prefix" ]; then
    echo "Could not locate conda env '$env_name' after create/update." >&2
    exit 1
fi

"$env_prefix/bin/python" -m pytest tests/test_installation.py "$@"
