#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python tests/test_pipeline.py
.venv/bin/python tests/test_recorded_frame.py
cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
echo "ci passed"
