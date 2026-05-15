#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

default_data_root="$HOME/Documents/datasets/create-pattern-detector"
data_root="${CP_SHARED_DATA_ROOT:-${CP_DATA_ROOT:-$default_data_root}}"
target="${CP_SCRAPED_DATASET:-$data_root/scraped}"
link_path="$repo_root/data/output/scraped"

if [[ ! -d "$target" ]]; then
  cat >&2 <<EOF
Shared scraped dataset not found:
  $target

Set CP_SHARED_DATA_ROOT or CP_SCRAPED_DATASET if your dataset lives elsewhere.
Expected examples:
  export CP_SHARED_DATA_ROOT="$default_data_root"
  export CP_SCRAPED_DATASET="$default_data_root/scraped"
EOF
  exit 1
fi

mkdir -p "$(dirname "$link_path")"

if [[ -L "$link_path" ]]; then
  current_target="$(readlink "$link_path")"
  if [[ "$current_target" == "$target" ]]; then
    echo "Shared scraped dataset already linked:"
    echo "  $link_path -> $target"
    exit 0
  fi

  rm "$link_path"
elif [[ -e "$link_path" ]]; then
  # Only remove an empty placeholder directory, or one containing macOS metadata.
  non_metadata_count="$(
    find "$link_path" -mindepth 1 \
      ! -name '.DS_Store' \
      ! -name 'Icon?' \
      -print -quit | wc -l | tr -d ' '
  )"

  if [[ "$non_metadata_count" != "0" ]]; then
    cat >&2 <<EOF
Refusing to replace non-empty scraped data path:
  $link_path

Move or back up that directory first, then rerun this script.
EOF
    exit 1
  fi

  rm -rf "$link_path"
fi

ln -s "$target" "$link_path"

echo "Linked shared scraped dataset:"
echo "  $link_path -> $target"

