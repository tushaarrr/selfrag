#!/usr/bin/env bash
# Phase 2 pilot on a free Kaggle notebook (Accelerator: GPU T4 x2, Internet: on). From a code cell:
#   !git -C selfrag pull -q || git clone -q https://github.com/tushaarrr/selfrag
#   !bash selfrag/kaggle_phase2.sh
# Labels the 1,952-example pilot with an open teacher and reports agreement with Self-RAG's labels.
set -euo pipefail
cd "$(dirname "$0")"
# Pressing Stop leaves the labeler and its tensor-parallel worker holding both GPUs; clear them first.
P='label_teacher.py label|/tmp/venv2/bin/python'; pkill -f "$P" && while pgrep -f "$P" >/dev/null; do sleep 2; done || true
MODEL=${MODEL:-Qwen/Qwen3-32B-AWQ}                     # fallback: MODEL=Qwen/Qwen3-14B-AWQ (about 2x faster)
OUT=${OUT:-runs/phase2/$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
HOURS=${HOURS:-11}                                      # Kaggle kills a session at 12h

pip install -q uv
rm -rf /tmp/venv2                                       # separate from Phase 1's /tmp/venv (vLLM 0.2.6)
uv venv -q --python 3.11 /tmp/venv2
/tmp/venv2/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
# --python, not VIRTUAL_ENV: Kaggle sets UV_SYSTEM_PYTHON (see kaggle_phase1.sh)
uv pip install -q --python /tmp/venv2/bin/python -r requirements-teacher.txt pytest
export PATH=/tmp/venv2/bin:$PATH HF_HOME=/tmp/hf          # keep the 19GB of weights out of /kaggle/working
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0                     # hf-xet 1.1.5 would also keep a 10GB chunk cache
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
./setup_data.sh >/dev/null
python -m pytest -q -p no:cacheprovider test_phase2.py  # includes the pinned pilot md5
python critic_data.py pilot --out runs/phase2/pilot.jsonl | tail -1

timeout 3600 huggingface-cli download "$MODEL" >/dev/null  # visible progress, capped, before the long timeout
rc=0
timeout $((HOURS * 3600)) python label_teacher.py label --examples runs/phase2/pilot.jsonl \
  --out "$OUT/pilot.labels.jsonl" --model "$MODEL" || rc=$?
if [ -f "$OUT/pilot.labels.jsonl" ]; then python label_teacher.py report --out "$OUT/pilot.labels.jsonl"; fi
zip -qr /kaggle/working/phase2.zip runs/phase2 && echo "saved /kaggle/working/phase2.zip; download it from the Output panel now"
if [ "$rc" -eq 124 ]; then echo "time limit hit; run again to resume"; elif [ "$rc" -ne 0 ]; then exit "$rc"; fi
