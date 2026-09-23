# Self-RAG reproduction: working notes

Follows `PLAN-revised.md`. Inputs are symlinked from `../self-rag` (`data/`, `reference/`, `paper/`, `docs/`); nothing there is edited.

| File | What |
|---|---|
| `eval_selfrag.py` | Phase 1/4 harness: `generate` (vLLM 0.2.6, GPU) and `score` (CPU, offline) |
| `train_generator.py` | Phase 3: LoRA trainer, reflection-token accuracy (acceptance checks), merge for vLLM |
| `test_*.py` | CPU-only. They run the reference's own code as an oracle (`pytest -q`, about 20 s) |
| `requirements.in` / `requirements.txt` | Pinned environment for eval and training (Linux, py3.10, CUDA 12.1 wheels) |

## Phase 0: environment (decided)

**One pinned environment:** vLLM 0.2.6, torch 2.1.2, transformers 4.36.2, peft 0.7.1, xformers 0.0.23.post1. It is resolved in `requirements.txt` for linux/py3.10. Training uses the same env, so train and eval tokenise with the same transformers code.

What I checked in the vLLM **source** (not run: no GPU here):
- **0.2.6 works with the reference as written:**
  - There is no cap on `logprobs`: 32016, the full vocab, is accepted.
  - Values are plain floats.
  - Temperature 0 is replaced by 1.0 before `log_softmax`, and with top_p=1 nothing is masked, so logprobs are raw model log-probs.
  - The sampled token is always in the dict.
  - `token_ids` and `cumulative_logprob` include the final EOS.
  - `LLM.generate` returns outputs in prompt order.
  - Defaults are `skip_special_tokens=True` and `spaces_between_special_tokens=True`.
- **The latest vLLM (0.30) breaks the reference:**
  - `max_logprobs` defaults to 20, and a larger request raises an error.
  - Values are `Logprob` objects, so `float(...)` fails.
  - Logprobs are raw by default (`logprobs_mode`).
  - A port would need `max_logprobs=-1`, `.logprob`, and ideally `logprob_token_ids` for the 16 reflection ids.
- The released tokenizer's `tokenizer.model` sha256 is `9e556afd…d347`. That is Llama-2's as I recall it; check it against the base download on the GPU box.

## Pre-registration (written 2026-09-22, before any Phase 1 run)

1. **PopQA, retrieve formula.**
   - The as-released formula `lp(Ret)/(lp(Ret)+lp(NoRet)) > 0.2` retrieves when p(Ret) < 0.724 (taking p(Ret)+p(NoRet)≈1). The paper formula retrieves when p(Ret) > 0.2.
   - So they don't simply "run opposite" (#5): they agree for 0.2 < p < 0.72 and disagree at both ends.
   - The paper's 50k ablation (Fig. 3a) has greedy "hard constraints" at 28.3, close to no-retrieval at 24.7. That means the model rarely makes [Retrieval] its argmax on PopQA, so p(Ret) is mostly low.
   - **Prediction:**
     - The as-released formula retrieves on ≥ 85% of PopQA items and lands within 2 points of 54.9.
     - The paper formula retrieves on clearly fewer items (≤ 60%) and scores ≥ 8 points lower.
   - Confidence: about 65%.
2. **ARC, match minus strict.**
   - Self-RAG trains on bare-letter answers (`[No Retrieval]B[Utility:5]`), and vLLM strips the reflection tokens from the text. So most predictions will be a bare `"C"`, and substring leniency should add close to 0.
   - Separately, 22/1,172 items have numeric gold keys (`"1"`-`"4"`) while the prompt relabels the options A-D. `match` can never score those; strict maps them.
   - **Prediction:** match − strict = **−1.3 points** (range −2.0 to +1.0). None of the 67.3 comes from leniency.
3. **Our 30k LoRA generator vs the released 7B** (anchors: the paper's 50k full-FT model is −9.4 on PopQA and +1.1 on PubHealth; PopQA is the most data-sensitive task, Fig. 4). **Prediction:**
   - PopQA: **−12** (−20 to −6)
   - TriviaQA: **−6** (−12 to −2)
   - ARC: **−4** (−10 to +2)
   - PubHealth: **−1** (−5 to +3)
4. **PubHealth barely depends on retrieval** (50k ablation: no-retrieval 73.0 vs adaptive 73.5). The as-released and paper formulas land within 2 points of each other on PubHealth.

## Decisions (defaults taken; say if you want them changed)

1. **Headline threshold:** as-released. It costs nothing to also report the paper formula, always-retrieve and never-retrieve, because `generate` produces both branches for every item and `score` prints all four.
2. **Training:** bf16 LoRA with trainable new-token rows (`modules_to_save=["embed_tokens","lm_head"]`, fp32 master weights).
3. **Critic (Phase 2):** skipped.

Weights: w_rel/w_sup/w_use = 1.0/1.0/**1.0**, as released. The paper says 0.5; `score --w_use 0.5` rescoring is free.

## Deviations from the reference

Eval (outputs identical by construction, checked by `test_eval_selfrag.py` against the reference `main()`):
- **Same retrieve decision, 1 token instead of 100:** the first call uses `max_tokens=1`. The reference only reads `logprobs[0]` from it, and that call's text never reaches the answer. This removes 100 full-vocab dicts per item.
- **Both branches generated for every item:** the no-retrieval continuation and the retrieval branch are generated for every item, then decided offline. That is about 9% more tokens.
- **Batched across items**, in `--chunk` items per vLLM call. fp16 batching can flip rare near-tie greedy tokens.
- **Empty predictions:** the reference crashes on an empty prediction (`run_short_form.py:331`). We continue and set `empty_pred`.

Training (`train_generator.py`):

| Setting | Reference | Ours |
|---|---|---|
| Method | Full fine-tune, ZeRO-3, 4 GPUs | bf16 LoRA r16/α32/dropout 0.05 on q,k,v,o,gate,up,down, plus full trainable copies of embed_tokens and lm_head; 1 GPU |
| Learning rate | 2e-5 | 1e-4 or 2e-4, picked by the pilot on held-out reflection accuracy |
| New-token init | HF normal init | Mean init |
| Batch | 1 × 32 × 4 GPUs | 8 × 16 with length grouping (loss is token-weighted within a micro-batch) |
| Data | 145,619 rows | Seeded 30k subset; the disjoint 1% held-out split is written to `split.json` |
| Attention | flash-attn | sdpa (numerics only) |

- **Kept on purpose:** the paragraph-mask bug (#13), which deviates from the paper but is faithful to the reference. So is the source-mask off-by-one (`labels[:src_len-1]`). Both are verified token for token against the reference's own `encode_with_prompt_completion_format` on 300 real rows.
- **Tokenizer:** the released one, loaded as the slow tokenizer as in the reference. Its ids are asserted as 32000-32015.
- **Caveat:** the released 7B trained on all rows, our held-out rows included. So its held-out reflection accuracy is an upper reference, not a fair held-out score.

## Findings while building

- **Stripped tokens:** the reflection tokens are `additional_special_tokens`, and vLLM's `skip_special_tokens=True` strips them from `.text`. The reference's control-token postprocessing is therefore a no-op, and `match` can't be fooled by `[Continue to Use Evidence]`.
- **ARC gold keys:** 22 items have numeric gold keys (see prediction 2). 3 items have an option E that the reference drops; none of them has E as the answer.
- **Empty `input`:** every one of the 145,619 `train.jsonl` rows has an empty `input`, so the reference's `prompt_input` template is never used.
- **Collator bug (ours, now fixed):** `DataCollatorForSeq2Seq` pads `labels` by writing back into the feature dict. With a list dataset, epoch 2 then crashes on a length mismatch. The tiny-model test caught it; the collator now gets copies.

## Where to run (researched 2026-09-22; T4 figures are estimates, not measurements)

- **This Mac (M4, 24GB, 16GB free disk):**
  - vLLM 0.2.6 is Linux + CUDA only. The eval could only run through a llama.cpp or MLX port, at 2-6 days, with logprobs that differ from the reference setup.
  - Training would take 5-15 days.
  - CPU-only is slower again: torch 2.1 has no fp16 CPU matmul, and fp32 weights (27GB) exceed RAM.
  - The disk can't hold the model plus a conversion.
- **Kaggle free (2× T4):**
  - Eval only: `--tp 2`, dtype half, roughly 5-15h across 1-2 resumed 12h sessions (estimate).
  - The image is Python 3.12, so create a py3.11 env for vllm==0.2.6.
  - Training doesn't fit: T4 has no bf16 and too little memory.
- **Rented A100 80GB:**
  - Eval about 1.5-4h; training about 3-6h.
  - Price: Vast ~$0.78-0.85/h median, RunPod $1.59/h (checked on the day).

## GPU runbook (1 × A100 80GB)

```bash
rsync -aL --exclude .venv --exclude data/hf/sms1097 --exclude data/eval_data.zip ./ BOX:selfrag/   # -L resolves the symlinks
python3.10 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pytest -q

# Phase 1: pilot first, then decide
python eval_selfrag.py generate --task popqa --out runs/released --limit 50    # prints s/item and projected hours
python eval_selfrag.py score    --task popqa --out runs/released              # sanity check on 50 items
python eval_selfrag.py generate --task popqa --out runs/released              # resumes; then score
#   then pubhealth, arc, triviaqa. Stop rule: >2 pts off (released formula) = harness bug.
#   Except TriviaQA: if it's off, run --task triviaqa_plain before calling it a bug.
#   --chunk 32 bounds host RAM for the logprob dicts. Lower it if the host swaps.

# Phase 3 (HF token with Llama-2 access)
python train_generator.py --out runs/released_reflect --eval_model selfrag/selfrag_llama2_7b
python train_generator.py --out runs/pilot_1e-4 --lr 1e-4 --max_steps 100
python train_generator.py --out runs/pilot_2e-4 --lr 2e-4 --max_steps 100   # pick on reflection_acc.json, per class
python train_generator.py --out runs/gen30k --lr <picked> --merge           # stops itself if >1.5x the 6.4h budget

# Phase 4
python eval_selfrag.py generate --task <t> --model runs/gen30k/merged --out runs/ours   # then score
```

Not built yet: the Phase 4 paired stats (McNemar and bootstrap over the `*.scored.jsonl` files, keyed by `idx`). They're about 20 lines, to write once both runs exist.
