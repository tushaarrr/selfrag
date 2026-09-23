# Phase 1 results: released Self-RAG 7B, all four short-form tasks

Run on 2026-09-23 on a free Kaggle notebook (2× T4, fp16, vLLM 0.2.6, tensor parallel 2), using `kaggle_phase1.sh` at commit `10f5828`. Settings match the reference README: threshold 0.2, weights 1.0 / 1.0 / 1.0, seqscore on, and ndocs 10 for PopQA and TriviaQA, 5 for PubHealth and ARC. Wall clock was about 6.5 hours.

## Headline: the released checkpoint reproduces the paper

| Task | n | Ours (as released) | Paper | Diff | 95% margin |
|---|---|---|---|---|---|
| PopQA | 1,399 | **55.0** | 54.9 | +0.1 | ±2.6 |
| TriviaQA | 7,313 | **66.2** | 66.4 | −0.2 | ±1.1 |
| PubHealth | 987 | **71.9** | 72.4 | −0.5 | ±2.8 |
| ARC-Challenge | 1,172 | **67.3** | 67.3 | 0.0 | ±2.7 |

Every task falls well inside its margin. Scores use the reference `match` metric, the same substring test as `metrics.py:83-87`.

## The four retrieval policies

All four policies are scored from the same single generation pass (see `eval_selfrag.py`). "Ret." is the share of questions where retrieval happened.

| Task | As released: ret. / acc | Paper formula: ret. / acc | Always: acc | Never: acc |
|---|---|---|---|---|
| PopQA | 100% / 55.0 | 89.6% / 54.1 | 55.0 | 28.7 |
| TriviaQA | 100% / 66.2 | 95.1% / 65.6 | 66.2 | 50.3 |
| PubHealth | 100% / 71.9 | 98.1% / 71.6 | 71.9 | 70.1 |
| ARC-Challenge | 100% / 67.3 | 12.2% / 67.0 | 67.3 | —* |

\*ARC's "never" row was cut off in the log excerpt; it is in `runs/released/summary.jsonl`.

## Findings

1. **"Adaptive" retrieval never skipped retrieval.** Under the released threshold code, which divides log-probabilities (`run_short_form.py:80`), the model retrieved on **100% of questions in all four tasks**. Its scores are identical to always-retrieve. So the paper's adaptive-retrieval numbers are in effect always-retrieve numbers.
2. **The paper's own formula saves retrieval at almost no cost.** Applying 0.2 to probabilities, as the paper describes, skips retrieval on 5-10% of open-domain questions and on **88% of ARC questions**. Accuracy drops by at most 0.9 points on any task.
3. **Retrieval matters very unevenly.** It adds +26.3 points on PopQA and +15.9 on TriviaQA, but only +1.8 on PubHealth.
4. **Substring leniency doesn't inflate ARC.** Strict letter accuracy is 68.8, so `match` − strict = **−1.5**. The lenient metric actually *loses* points on the 22 questions with numeric gold keys.
5. **TriviaQA 66.4 comes from the 7,313-item `*_w_gs` file**, the one with Google-search passages at positions 5-9, not the 11,313-item plain file. The plan had flagged this as undeterminable from the repo (PLAN-revised #15).
6. **So where do the reported reproduction gaps come from?** The released checkpoint reproduces under the pinned 2023 stack. The gaps reported by independent reruns ([FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)) and by retraining attempts ([#57](https://github.com/AkariAsai/self-rag/issues/57), [#71](https://github.com/AkariAsai/self-rag/issues/71)) therefore come from differences in eval setup (metric, passages, inference stack) or from training, not from the checkpoint itself. Phase 3 tests the training side.

## Pre-registered predictions (written before the run, in NOTES.md)

| # | Prediction | Outcome |
|---|---|---|
| 1a | As released: retrieves on ≥85% of PopQA, lands within 2 points of 54.9 | ✅ 100%, +0.1 |
| 1b | Paper formula: retrieves on ≤60% of PopQA and scores ≥8 points lower | ❌ 89.6%, only −0.9 |
| 2 | ARC match − strict = −1.3 (range −2.0 to +1.0) | ✅ −1.5 |
| 3 | Our 30k LoRA generator vs released, per task | Pending (Phase 3) |
| 4 | PubHealth: the two formulas land within 2 points | ✅ 71.9 vs 71.6 |

Prediction 1b was wrong. On PopQA the model's p(Retrieve) is mostly in the 0.2-0.72 range where the two formulas agree; the paper's 50k-model ablation had suggested otherwise.
