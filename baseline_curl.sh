#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

request_body="$(
  python3 - "${script_dir}/prompt.md" <<'PY'
import json
import pathlib
import sys

prompt = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
print(json.dumps({
    "model": "Qwen/Qwen2.5-0.5B",
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0.9,
}))
PY
)"

curl --fail-with-body --silent --show-error \
  -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  --data "${request_body}"
printf '\n'
