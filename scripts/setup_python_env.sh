#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CP_PYTHON:-python}"
venv_root="${CP_PYTHON_ENV_ROOT:-$HOME/.cache/create-pattern-detector/venvs}"

force_install=0
replace_local=0
adopt_local=0

usage() {
  cat <<EOF
Usage: scripts/setup_python_env.sh [--force] [--replace-local] [--adopt-local]

Creates or reuses a shared Python virtual environment, then links this worktree's
.venv to it. This keeps future worktrees from reinstalling the full dependency
stack.

Environment overrides:
  CP_PYTHON=/path/to/python
  CP_PYTHON_ENV_ROOT=/path/to/shared/venvs
  CP_PYTHON_VENV=/path/to/exact/venv

Options:
  --force          Reinstall dependencies even if the environment stamp matches.
  --replace-local  Replace an existing non-symlink .venv in this worktree.
  --adopt-local    Move an existing non-symlink .venv into the shared cache when
                   the target shared venv does not already exist.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      force_install=1
      ;;
    --replace-local)
      replace_local=1
      ;;
    --adopt-local)
      adopt_local=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v "$python_bin" >/dev/null 2>&1; then
  cat >&2 <<EOF
Python executable not found: $python_bin

Install the version in .python-version or set CP_PYTHON=/path/to/python.
EOF
  exit 1
fi

python_version="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"

venv_path="${CP_PYTHON_VENV:-$venv_root/py${python_version}-dev}"
stamp_path="$venv_path/.cp-detector-env.stamp"
local_link="$repo_root/.venv"

env_stamp="$(
  {
    printf 'python=%s\n' "$python_version"
    shasum -a 256 "$repo_root/.python-version" "$repo_root/pyproject.toml"
  } | shasum -a 256 | awk '{print $1}'
)"

mkdir -p "$(dirname "$venv_path")"

if [[ "$adopt_local" == "1" && -d "$local_link" && ! -L "$local_link" && ! -e "$venv_path" ]]; then
  mv "$local_link" "$venv_path"
  force_install=1
  echo "Adopted existing local environment:"
  echo "  $venv_path"
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  echo "Creating shared Python environment:"
  echo "  $venv_path"
  "$python_bin" -m venv "$venv_path"
fi

installed_stamp=""
if [[ -f "$stamp_path" ]]; then
  installed_stamp="$(cat "$stamp_path")"
fi

if [[ "$force_install" == "1" || "$installed_stamp" != "$env_stamp" ]]; then
  echo "Installing project dependencies into shared environment:"
  echo "  $venv_path"
  "$venv_path/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
  "$venv_path/bin/python" -m pip install -e "$repo_root[dev]"
  printf '%s\n' "$env_stamp" > "$stamp_path"
else
  echo "Shared Python dependencies are already current:"
  echo "  $venv_path"
  # Keep console scripts and editable metadata pointed at the current worktree.
  "$venv_path/bin/python" -m pip install --no-deps -e "$repo_root" >/dev/null
fi

if [[ -L "$local_link" ]]; then
  current_target="$(readlink "$local_link")"
  if [[ "$current_target" != "$venv_path" ]]; then
    rm "$local_link"
    ln -s "$venv_path" "$local_link"
  fi
elif [[ -e "$local_link" ]]; then
  if [[ "$replace_local" == "1" ]]; then
    rm -rf "$local_link"
    ln -s "$venv_path" "$local_link"
  else
    cat >&2 <<EOF
Refusing to replace existing non-symlink .venv:
  $local_link

Re-run with --replace-local to replace it, or --adopt-local to move it into the
shared cache if the target shared venv does not exist.
EOF
    exit 1
  fi
else
  ln -s "$venv_path" "$local_link"
fi

echo "Python environment ready:"
echo "  .venv -> $venv_path"
echo
echo "Use:"
echo "  .venv/bin/python -m pytest"
