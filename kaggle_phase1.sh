#!/usr/bin/env bash
# Phase 1 on a free Kaggle notebook (Accelerator: GPU T4 x2, Internet: on). From a code cell:
#   !git clone -q https://github.com/tushaarrr/selfrag && bash selfrag/kaggle_phase1.sh
# Resume in a later session by attaching the previous version's output and setting PREV (README: "Run on Kaggle").
set -euo pipefail
cd "$(dirname "$0")"
TASKS=${TASKS:-"popqa pubhealth arc triviaqa"}
HOURS=${HOURS:-11}   # Kaggle kills a session at 12h; stop early so /kaggle/working is saved
PREV=${PREV:-}       # e.g. /kaggle/input/<previous-notebook>/selfrag/runs
LIMIT=${LIMIT:-}     # e.g. 50 for the pilot
deadline=$((SECONDS + HOURS * 3600))

pip install -q uv
uv venv -q --python 3.10 /tmp/venv   # Kaggle's Python is 3.12; vllm 0.2.6 ships wheels up to 3.11
VIRTUAL_ENV=/tmp/venv uv pip install -q -r requirements.txt pytest
export PATH=/tmp/venv/bin:$PATH HF_HOME=/tmp/hf   # keep the 13.5GB of weights out of /kaggle/working
./setup_data.sh >/dev/null
python -m pytest -q -p no:cacheprovider test_eval_selfrag.py
if [ -n "$PREV" ]; then mkdir -p runs && cp -rn "$PREV/." runs/; fi

for t in $TASKS; do
  left=$((deadline - SECONDS))
  if [ "$left" -le 900 ]; then echo "out of time before $t; run again with PREV set to resume"; break; fi
  rc=0
  timeout "$left" python eval_selfrag.py generate --task "$t" --out runs/released --tp 2 ${LIMIT:+--limit "$LIMIT"} || rc=$?
  if [ -f "runs/released/$t.gen.jsonl" ]; then python eval_selfrag.py score --task "$t" --out runs/released; fi
  if [ "$rc" -eq 124 ]; then echo "time limit hit during $t; run again with PREV set to resume"; break; fi
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
