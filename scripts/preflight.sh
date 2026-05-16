#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="$ROOT/venv/bin/python"
fi
if [[ -n "${RUFF:-}" ]]; then
  RUFF_BIN="$RUFF"
elif [[ -x "$ROOT/.venv/bin/ruff" ]]; then
  RUFF_BIN="$ROOT/.venv/bin/ruff"
else
  RUFF_BIN="$ROOT/venv/bin/ruff"
fi
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PATH="$PYTHON_BIN_DIR:$PATH"

section() {
  printf '\n==> %s\n' "$*"
}

run() {
  section "$*"
  "$@"
}

require_clean_git() {
  section "Checking git status"
  if [[ "${D2C_ALLOW_DIRTY:-0}" == "1" ]]; then
    git -C "$ROOT" status --short --branch
    return
  fi

  local status
  status="$(git -C "$ROOT" status --porcelain)"
  if [[ -n "$status" ]]; then
    git -C "$ROOT" status --short --branch
    printf '\nPreflight requires a clean worktree. Set D2C_ALLOW_DIRTY=1 only for local debugging.\n' >&2
    exit 1
  fi
  git -C "$ROOT" status --short --branch
}

scan_retired_task_name() {
  section "Scanning tracked files for retired benchmark references"
  local retired_name
  retired_name="$(printf '%s%s' 'Huma' 'noid')"
  if git -C "$ROOT" grep -I -n -i "$retired_name" -- .; then
    printf '\nTracked retired benchmark references found.\n' >&2
    exit 1
  fi
}

scan_secret_patterns() {
  section "Scanning tracked files for secret-like values"
  local pattern
  pattern='(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9_]{36,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY)'
  if git -C "$ROOT" grep -I -n -E "$pattern" -- .; then
    printf '\nSecret-like tracked values found.\n' >&2
    exit 1
  fi
}

check_tmp_imports_and_cli() {
  section "Checking installed imports and console help from /tmp"
  (
    cd /tmp
    "$PYTHON_BIN" - <<'PY'
import src.debate
import utils.make_env
import envs.hopper
PY
    for cmd in \
      d2c-debate \
      d2c-train-xml \
      d2c-render \
      d2c-render-policy \
      d2c-baselines \
      d2c-plot
    do
      "$cmd" --help >/dev/null
    done
  )
}

check_python_repo() {
  run "$RUFF_BIN" check "$ROOT"
  run "$PYTHON_BIN" -m compileall -q "$ROOT/src" "$ROOT/envs" "$ROOT/utils" "$ROOT/scripts" "$ROOT/baselines" "$ROOT/render.py"
  section "Running CPU unittest smoke suite"
  (
    cd "$ROOT"
    PYTHONPATH=src:. "$PYTHON_BIN" -m unittest discover -s tests
  )
  section "Running no-key Hydra config check"
  (
    cd "$ROOT"
    PYTHONPATH=src:. "$PYTHON_BIN" src/debate.py env=hopper debate.rounds=0 debate.enable_judges=false
  )
  check_tmp_imports_and_cli
}

check_archive() {
  section "Checking source archive staging tree"
  local stage_parent stage_dir
  stage_parent="$(mktemp -d)"
  stage_dir="$stage_parent/debate2create"
  mkdir -p "$stage_dir"

  cleanup_stage() {
    rm -rf "$stage_parent"
  }
  trap cleanup_stage RETURN

  git -C "$ROOT" archive --format=tar HEAD | tar -xf - -C "$stage_dir"
  (
    cd "$stage_dir"
    git init -q
    git add -A
    local retired_name
    retired_name="$(printf '%s%s' 'Huma' 'noid')"
    if git grep -I -n -i "$retired_name" -- .; then
      printf '\nRetired benchmark references found in source archive.\n' >&2
      exit 1
    fi
    local pattern
    pattern='(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9_]{36,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY)'
    if git grep -I -n -E "$pattern" -- .; then
      printf '\nSecret-like values found in source archive.\n' >&2
      exit 1
    fi
    "$PYTHON_BIN" -m compileall -q src envs utils scripts baselines render.py
    PYTHONPATH=src:. "$PYTHON_BIN" -m unittest discover -s tests
  )
  cleanup_stage
  trap - RETURN
}

main() {
  require_clean_git
  scan_retired_task_name
  scan_secret_patterns
  check_python_repo
  check_archive
  require_clean_git
  section "Preflight passed"
}

main "$@"
