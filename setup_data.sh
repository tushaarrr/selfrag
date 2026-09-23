#!/usr/bin/env bash
# Fetch the inputs into data/ and reference/, then verify them against SHA256SUMS. About 1.2GB, no model weights.
# Needs git, unzip and huggingface-cli (from the pinned env: pip install -r requirements.txt).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/hf reference

[ -d reference/self-rag ] || { git clone -q https://github.com/AkariAsai/self-rag reference/self-rag &&
  git -C reference/self-rag checkout -q 1fcdc42; }

# The README's Google Drive link is dead; this HF mirror matches two independent copies (PLAN-revised #2).
[ -d data/eval_data ] || { huggingface-cli download JIM-Zhangxw/eval_self_rag eval_data.zip --repo-type dataset \
  --local-dir data && unzip -q data/eval_data.zip -d data; }
[ -f data/hf/selfrag_train_data/train.jsonl ] || huggingface-cli download selfrag/selfrag_train_data train.jsonl \
  --repo-type dataset --local-dir data/hf/selfrag_train_data
# tokenizer and configs only; vLLM downloads the weights itself
[ -f data/hf/selfrag_llama2_7b/tokenizer.model ] || huggingface-cli download selfrag/selfrag_llama2_7b \
  added_tokens.json config.json generation_config.json special_tokens_map.json tokenizer.model tokenizer_config.json \
  --local-dir data/hf/selfrag_llama2_7b

(cd data && shasum -a 256 -c ../SHA256SUMS)
