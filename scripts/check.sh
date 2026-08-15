#!/usr/bin/env bash
# Runs the same gates as .github/workflows/checks.yml and scripts/check.ps1.
set -uo pipefail

run_check() {
    local name=$1
    shift
    printf '== %s\n' "$name"
    if ! "$@"; then
        printf '== %s FAILED (exit %d)\n' "$name" "$?" >&2
        exit 1
    fi
}

run_check compileall  python -m compileall sidewalk_ai -q
run_check ruff        python -m ruff check sidewalk_ai
run_check black       python -m black --check sidewalk_ai
run_check pytest      python -m pytest
run_check diagnostics python -m sidewalk_ai.diagnostics

printf 'All checks passed\n'
