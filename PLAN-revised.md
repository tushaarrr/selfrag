# Self-RAG reproduction — revised plan (v2)

Replaces the phase structure of `docs/PLAN.md`. The original is left unedited. Every change below is backed by
code, data or paper evidence gathered on 2026-09-22 (the reference is at `reference/self-rag` @ 1fcdc42).
Evidence paths are relative to `reference/self-rag/` unless noted.

## What changed and why

| # | PLAN.md said | Reality | Evidence |
|---|---|---|---|
| 1 | Phase 1 uses the 9GB demo corpus | Not needed. Every eval file ships its passages in `ctxs`, and the short-form script never calls a retriever. The demo-corpus link is dead anyway. | `retrieval_lm/run_short_form.py:203-207`; `data/eval_data/*` (`ctxs`: 20-25 per item); `download_demo_corpus.sh` Drive id → 404 |
| 2 | Eval data comes from the README | README Drive link is dead (404, checked against a working control file). Use the HF mirror `JIM-Zhangxw/eval_self_rag`: 7 of 8 files are sha256-identical to two other independent mirrors, and the zip's internal file dates are 2023-10-18. | `data/eval_data/` |
| 3 | One command for all four tasks | Per-task settings differ (see the Phase 1 table). The plan's command also drops `--use_seqscore`, which changes the candidate score. | README; `run_short_form.py:152-157` |
| 4 | Run `run_short_form.py` | As shipped it should crash on the first item: it passes `max_depth=` to a function that doesn't accept it. This has been true since the first commit. | `run_short_form.py:315` vs `:51-54` |
| 5 | Threshold 0.2 decides adaptive retrieval | Short-form applies the threshold to a ratio of **log-probs**: `lp(Ret)/(lp(Ret)+lp(NoRet))`. That runs opposite to the paper's probability ratio. Long-form uses `np.exp` first (correct). | `run_short_form.py:78-81` vs `run_long_form_static.py:186-189` |
| 6 | `--metric match` = accuracy | Raw, case-sensitive substring test with no normalisation. The ARC gold is a single letter (`"C"`), so any capital C in the output scores as correct. | `retrieval_lm/metrics.py:83-87`; `run_short_form.py:242` |
| 7 | Critic data: ~37k, downloadable | Not downloadable: the Drive link is 404 and HF has only the generator set. The released critic trained on **~24.5k examples for 2 epochs** (3,058 steps × eff. batch 16 = 48,953 samples = 0.702 samples/s × 69,734 s). | `data/hf/self_rag_critic/trainer_state.json` |
| 8 | Critic: 3 epochs via `train_special_tokens.py` | The released run used 2 epochs. The script should crash as shipped: a relative import (`:27`) and missing `PROMPT_DICT` keys (`:236`). | `data_creation/train_special_tokens.py` |
| 9 | Checkpoints: generator 7B/13B | A released critic `selfrag/self_rag_critic` also exists (~27GB fp32). It has **no** reflection special tokens (vocab 32001, `[PAD]` only). | `data/hf/self_rag_critic/config.json`, `added_tokens.json` |
| 10 | QLoRA substitutes for full FT | The generator adds 16 tokens (ids 32000-32015), and Llama2's `lm_head` is **untied** (`tie_word_embeddings: false`). The repo's own LoRA path has `modules_to_save` commented out, so new-token input and output rows stay at random init. The plan's LoRA config (q,k,v,o,gate,up,down) has the same hole. | `retrieval_lm/finetune.py:474-489`; `data/hf/selfrag_llama2_7b/config.json` |
| 11 | "Same lr 2e-5" under LoRA | 2e-5 is the full-FT lr (`script_finetune_7b.sh:26`). LoRA adapters usually need about 5-10× more. Pilot it, don't assume. | — |
| 12 | Phase 3: run our critic to augment the data | Augmenting from scratch needs the full Wikipedia index (~70GB, ~100GB RAM). `train.jsonl` is **already** critic-augmented with passages inline: 145,619 rows (the paper also says 145,619, not 150k). | `data/hf/selfrag_train_data/train.jsonl`; paper p.17 |
| 13 | Masking matches the paper | The paragraph-mask loop has no `break`, so it masks from the first `<paragraph>` to the **last** `</paragraph>`. That removes the loss on reflection tokens and text between passages in 22.8% of training rows (those with ≥2 paragraphs). | `finetune.py:271-284` |
| 14 | README: 8×A100 | Paper: 4×A100-80GB, batch 128, 3 epochs, ZeRO-3, max len 2048. The script matches the paper (4 GPUs × 1 × 32 accum). | paper p.19; `script_finetune_7b.sh:4-7` |
| 15 | PopQA and TriviaQA use Contriever passages | In `*_w_gs` files, positions 5-9 are **Google Programmable Search** passages (paper p.20). `--ndocs 10` always includes them. TriviaQA `w_gs` has 7,313 items, while the paper and the plain file have 11,313. | `data/eval_data/`; paper p.20 |
| 16 | Pre-register the Alpaca-RAG PubHealth drop and the 13B MAUVE drop | No phase runs Alpaca, 13B or ASQA, so neither prediction can ever be resolved. | PLAN.md phases |
| — | targets.csv numbers | All 5 rows × 4 tasks and both ASQA rows match paper Table 2. **CONFIRMED.** | paper text lines 452-466 |

## Phases (cost at $1.50/h, the plan's own rate: swap in your provider's)

| Phase | What | GPU-h (derived) | Cost | Standalone claim |
|---|---|---|---|---|
| 0 | Provenance + env pin (done except env) | 0 | $0 | "Inputs are hash-pinned" |
| 1 | Eval the released 7B, 4 short-form tasks, no index | 1-3 batched (pilot decides) | $2-5 | "Reproduced / didn't reproduce, with the mechanism logged" |
| 2 | ~~Critic retrain~~ → **optional**, off the critical path | 0 (or 3-5) | $0 (or ~$40-100 API + $5-8 GPU) | — |
| 3 | Train the generator on a 30k subset of the released `train.jsonl` | 4-8 (bf16 LoRA), +20% pilot | $7-15 | "Trained a Self-RAG generator on 1 GPU" |
| 4 | Ours vs released, paired | 1-2 | $2-3 | The finding |

Total is about **$11-23** of GPU without phase 2. The one-off downloads are the 13.5GB released 7B and the 13GB Llama-2-7b base.

---

## Phase 0 — provenance and environment (CPU, $0)

Done: the eval files and `train.jsonl` are hash-pinned (`data/SHA256SUMS`).

To do: **decide the inference stack before paying for a GPU.**
- The reference pins `vllm==0.2.6` and `transformers==4.36.2`. It asks for `logprobs=32016` (the full vocab) and calls `float(logprob)` on the results.
- On current vLLM, both likely break: requested logprobs are capped by `max_logprobs`, and logprob values are objects, not floats. Verify on the pinned version.
- Recommendation: a pinned container (vLLM 0.2.6 / torch 2.1 / CUDA 12.1) for the faithful run. Port later.

## Phase 1 — reproduce the eval with `selfrag/selfrag_llama2_7b`

Use your own harness, since the reference script crashes (#4). Mirror the reference logic line by line, and **batch across items**: the reference makes one `generate` call per item.

| Task | File | n | ndocs | max_new_tokens | task flag | Target |
|---|---|---|---|---|---|---|
| PopQA | `popqa_longtail_w_gs.jsonl` | 1,399 | 10 | 100 | — | 54.9 |
| TriviaQA | `triviaqa_test_w_gs.jsonl` | 7,313 | 10 | 100 | — | 66.4 |
| PubHealth | `health_claims_processed.jsonl` | 987 | 5 | 50 | `fever` | 72.4 |
| ARC-C | `arc_challenge_processed.jsonl` | 1,172 | 5 | 50 | `arc_c` | 67.3 |

Common flags: `--threshold 0.2 --use_groundness --use_utility --use_seqscore --dtype half`. ARC and PubHealth pass no `--mode`, and the default runs the adaptive branch.

Weights w_rel / w_sup / w_use are **1.0 / 1.0 / 1.0** as released: the argparse default for `--w_use` is 1.0 (`run_short_form.py:281-286`) and no README command overrides it. The paper (p.7) says 0.5. Log which one you use.

Log per item:
- Retrieve score under **both** formulas: as-released (log-prob ratio) and paper (prob ratio).
- The decision taken and the top-10 reflection-token probabilities.
- Every candidate's `final_score` and its components.
- The prediction, plus `match` exactly as `metrics.py:83-87`.

Also report a strict secondary metric: parsed letter for ARC, parsed true/false for PubHealth.

Budget arithmetic (upper bound: every item retrieves and every generation hits max tokens):
- PopQA: 1,399 × 11 × 100 = 1.54M output tokens.
- TriviaQA: 7,313 × 11 × 100 = 8.0M.
- PubHealth: 987 × 6 × 50 = 0.30M.
- ARC: 1,172 × 6 × 50 = 0.35M.
- Total ≈ 10.2M. Greedy answers stop early, so expect 30-50% of that (3-5M).
- At ~1.5-2.5k output tok/s (batched vLLM, 7B, A100), that is 0.5-2h plus prefill.
- Full-vocab logprob dicts are a large, unmeasured overhead. **Time 50 PopQA items first and extrapolate before running the rest.**

Stop rule: a task more than 2 points off means a harness bug, except TriviaQA. Which file produced 66.4 can't be determined from the repo (7,313 vs 11,313). If it's off, rerun on `triviaqa_test.jsonl` at ndocs 5 before calling it a bug.

## Phase 2 — critic (optional; decide before spending)

The generator data is already critic-augmented, so no later phase needs a new critic. Do this only if a "we retrained the critic" claim is worth it:
- **Labels:** regenerate them. The GPT-4 set isn't downloadable (#7). Expect about 25-30k calls, anchored on the 24.5k examples the released critic trained on, at ~0.5-0.9k input tokens per call. That is about $40-100 at current GPT-4-class list prices (check prices on the day) and about $0.4-1.1k at 2023 GPT-4 prices. The scripts default to `gpt-3.5-turbo` and the `openai==0.28` API, so they need porting.
- **Training:** use your own script (#8). LoRA/QLoRA is safe here because the critic adds no reflection special tokens (#9). Use 2 epochs to match the released run, or 3 as the README says, and record which.
- **Evaluation:** per-type confusion matrices against the held-out regenerated labels. The free alternative (agreement with labels inside `train.jsonl`) only measures how well you copy the released critic.

## Phase 3 — train the generator

- **Data:** a seeded 30k random subset of the released `train.jsonl`. No index, no critic run.
- **Method:** a bf16 base with LoRA r=16, α=32, on q, k, v, o, gate, up and down. QLoRA saves nothing for a 7B model on an 80GB card and adds a 4-bit confound.
- **New tokens:**
  - Add the 16 tokens in the released id order (`added_tokens.json`: 32000-32015), so eval token ids match.
  - Mean-initialise the new rows.
  - Make the **new rows trainable in both `embed_tokens` and `lm_head`**, using `modules_to_save=["embed_tokens","lm_head"]` or PEFT `trainable_token_indices` if your PEFT version supports it for an untied `lm_head`.
  - The `modules_to_save` route adds about 262M trainable params (2 × 32,016 × 4,096), about 3GB with optimizer state.
- **Learning rate:** a 2-arm pilot (1e-4 vs 2e-4, 100 optimizer steps each). Pick on held-out reflection-token accuracy, not loss.
- **Masking:** keep the reference behaviour (#13) for the faithful run and write it down as a deviation from the paper. Fixing it is an extension.
- **Cleanup:** drop the reference's debug dump (`finetune.py:518-530` writes every example's token ids to `processed.json` and prints each one).
- **Batching:** use micro-batch >1 with length grouping. The reference's batch 1 (mean 405 tokens) wastes the GPU.
- **Resilience:** resumable, checkpoint every 200 steps, manifest first.

Acceptance checks (these catch "loss looks fine, tokens broken"):
1. After 50 steps, the new-token rows in `embed_tokens` **and** `lm_head` have moved from init. Assert on the norm of the difference.
2. On a held-out 1% of `train.jsonl`, run teacher-forced top-1 accuracy at every Retrieve / IsRel / IsSup / IsUse position. Compare against the released 7B on the same rows.
3. Label skew to expect in the data: `[Utility:3]` 109 vs `[Utility:5]` 122,599; `[Relevant]` 159k vs `[Irrelevant]` 45k. Report per-class results, not a single average.

Budget arithmetic:
- Llama2-tokenised mean length is 405 tokens (sample of 4,000; p99 2,687; 2.4% over 2048 and truncated).
- 30k × 405 × 3 epochs ≈ 36.5M tokens.
- bf16 LoRA with gradient checkpointing costs ≈ 6N ≈ 40 GFLOP/token. At 20-35% MFU of 312 TFLOPS that is ~1.6-2.7k tok/s, so **3.8-6.4h**. QLoRA runs about 1.5-2× slower.
- Full set: 145,619 × 405 × 3 ≈ 177M tokens, so 18-31h. Ask before scaling up.
- These are derived, not measured. Measure tok/s over the first 200 steps and stop if the extrapolated time is more than 1.5× the estimate.

## Phase 4 — ours vs released

- Same harness, same threshold mode, same items.
- Stats: paired per-task McNemar and a bootstrap CI (as in the original plan).
- Add per-item Retrieve-decision agreement between the two models: that's the mechanism, not just the score.
- Cost: one more Phase 1 pass (1-2h).

## Pre-registration (replace the two unresolvable predictions)

Write these before Phase 1:
1. The released 7B's PopQA retrieval rate under the as-released threshold vs the paper formula, and which one reproduces 54.9.
2. ARC `match` minus strict-letter accuracy: how many points of 67.3 come from substring leniency.
3. Our 30k LoRA generator's gap to the released model per task, with a sign and a size.

## Decisions you own

1. **Headline threshold:** as-released (log-prob, likely what produced the paper numbers) or the paper formula. Recommendation: report as-released as the reproduction and the paper formula as a finding.
2. **Training method:** bf16 LoRA with trainable new-token rows (recommended), or QLoRA.
3. **Critic:** skip (recommended), or regenerate labels for about $40-100 plus porting work.

## Do not

- Download the demo corpus or the Wikipedia index. No phase needs them.
- Run any reference script as-is on a paid GPU (#4, #8, and the debug dump).
- LoRA-train the generator without trainable new-token rows (#10).
