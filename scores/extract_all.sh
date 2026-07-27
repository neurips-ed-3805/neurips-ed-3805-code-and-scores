#!/usr/bin/env bash
# to extract zip files, run:
# bash extract_all.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

command -v unzip >/dev/null 2>&1 || { echo "ERROR: 'unzip' is not installed." >&2; exit 1; }

echo "Extracting pointwise archives into: $DIR"
for z in *.zip; do
  echo "  unzip $z"
  unzip -o -q "$z"
done

for sub in likert pairwise temperature_effects; do
  [ -d "$sub" ] || continue
  echo "Extracting $sub archives into: $DIR/$sub"
  ( cd "$sub"
    for z in *.zip; do
      echo "  unzip $sub/$z"
      unzip -o -q "$z"
    done
  )
done
