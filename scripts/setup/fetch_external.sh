#!/usr/bin/env bash
# Fetch the READ-ONLY external sources at the commits this project was audited against, and put
# the ShuttleSet annotation CSVs where the loader expects them (data/raw/shuttleset/set).
#
#   scripts/setup/fetch_external.sh            # run from the project root
#   EXTERNAL_DIR=/some/path scripts/setup/fetch_external.sh
#
# Your own FORK of CoachAI+ is NOT handled here (docs/SETUP.md, step 5): it is a repo you edit.
set -euo pipefail

EXT="${EXTERNAL_DIR:-../external}"
mkdir -p "$EXT" data/raw/shuttleset

pin() {   # pin <name> <url> <full-sha>
  local name=$1 url=$2 sha=$3 dir="$EXT/$1"
  if [ ! -d "$dir/.git" ]; then git init -q "$dir"; git -C "$dir" remote add origin "$url"; fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "  $name @ $(git -C "$dir" rev-parse --short HEAD)"
}

echo "Fetching pinned sources into $EXT"
pin CoachAI-Projects https://github.com/wywyWang/CoachAI-Projects.git 45517f7d4cb936b03f3eabf939cc7959d39226fe

echo "Copying ShuttleSet annotations into data/raw/shuttleset/set"
rm -rf data/raw/shuttleset/set
cp -r "$EXT/CoachAI-Projects/ShuttleSet/set" data/raw/shuttleset/set
n=$(find data/raw/shuttleset/set -name 'set*.csv' | wc -l)
echo "  $n stroke CSV files (expected 104)"
[ "$n" -eq 104 ] || { echo "UNEXPECTED FILE COUNT: the upstream data changed. Re-run the audit before using it."; exit 1; }
echo "done."
