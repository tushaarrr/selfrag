# Where this fits: the RAG-critic landscape (checked 2026-09-22)

The question: is a small, open "RAG critic" worth building? It would answer, for any LLM, whether to retrieve, whether a passage is relevant, and whether each answer sentence is supported. Every claim below links to its source. Numbers are as reported by their authors unless noted.

## 1. Self-RAG itself: reproductions don't match the paper

- **Repo status:** the last code commit is `1fcdc42` (March 2024), and there are 67 open issues. The `max_depth` crash is [#39](https://github.com/AkariAsai/self-rag/issues/39) and has no reply. Current vLLM rejects its logprob requests ([#85](https://github.com/AkariAsai/self-rag/issues/85)).
- **Retraining gap:** retraining from the released data and script gives TriviaQA 50.3 vs 67.9 for the released checkpoint ([#57](https://github.com/AkariAsai/self-rag/issues/57)). A full reproduction got PopQA 50.8 and TriviaQA 58.0 ([#71](https://github.com/AkariAsai/self-rag/issues/71)). Baselines don't reproduce either ([#20](https://github.com/AkariAsai/self-rag/issues/20), [#97](https://github.com/AkariAsai/self-rag/issues/97)).
- **Independent reruns:**
  - [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) scores the released 7B *below* standard RAG (TriviaQA 38.2 vs 58.9 EM).
  - [RAGLAB](https://arxiv.org/html/2408.11381) finds that at 8B, Self-RAG "did not significantly surpass other RAG algorithms".
- **What nobody has published:** why the numbers don't match. This repo's harness separates the candidate causes: the threshold formula, the metric, the vLLM version, and the training-mask and LoRA problems. **Explaining that gap is the most immediate useful contribution.**

## 2. Is this passage relevant / does it answer the question?

| Tool | Size | License | Notes |
|---|---|---|---|
| [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B / 4B / 8B | Apache-2.0 | Outputs P(yes); the 0.6B scores MTEB-R 65.8 |
| bge-reranker-v2-m3, mxbai-rerank-v2 | ~0.6B | Apache-2.0 | [comparison](https://futureagi.com/blog/best-rerankers-for-rag-2026/) |
| [CRAG evaluator](https://github.com/HuskyInSalt/CRAG) | T5 | open | Relevance only; last update 2024-10 |
| [Tiny-Critic RAG](https://arxiv.org/html/2603.00846) | Qwen3-1.7B LoRA | no weights found | Binary pass/fail relevance gate |

**Self-RAG-style critics that already exist:**
- **[sms1097's four DistilBERT classifiers](https://huggingface.co/sms1097/support_model)** (67M each, MIT, Feb 2024). One classifier per token (Retrieve / IsRel / IsSup / IsUse), trained on a [flattened copy](https://huggingface.co/datasets/sms1097/self_rag_tokens_train_data) of the Self-RAG data. They report accuracy only on their own split and have about 1k downloads.
- **The paper's FLAN-3B critic was never released.** Agreement with GPT-4 labels (Retrieve / IsSup / IsRel / IsUse): 85.6 / 73.1 / 82.0 / 72.1, vs 93.8 / 93.5 / 80.2 / 73.5 for the 7B ([paper appendix](https://arxiv.org/pdf/2310.11511)).
- **Small RAG critics with *other* label sets:** [RAG-Critic-3B](https://huggingface.co/dongguanting/RAG-Critic-3B) (error taxonomy; its dataset is CC BY-NC), Tiny-Critic (no weights), CRITIC-R1.
- **These are baselines any new critic must beat.**

**Verdict:** relevance ranking is well served, for free. "Is this context *sufficient* to answer the question?" is not.

## 3. Is each sentence supported by the passages?

| Tool | Size | License | LLM-AggreFact* | Notes |
|---|---|---|---|---|
| [Bespoke-MiniCheck-7B](https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B) | 7B | **CC BY-NC** | 77.4 (#1) | Non-commercial |
| [Granite Guardian 3.3](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b) | 8B | Apache-2.0 | 76.5 | Also checks context relevance and answer relevance |
| [FactCG-DeBERTa-L](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large) | 0.4B | MIT | 75.6 | Best ≤2B |
| [MiniCheck-Flan-T5-L](https://github.com/Liyan06/MiniCheck) | 0.8B | Apache-2.0 | 75.0 | Last news Sept 2024 |
| [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model) | 0.1B | open | — | Runs on CPU (~1.5 s per 2k tokens); built into [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) |
| [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) | 17M–2B | MIT | — | Flags unsupported spans; multilingual; active (v0.2.2, July 2026) |
| [granitelib-rag](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) | LoRAs on Granite 4 | Apache-2.0 | — | Per-sentence risk, answerability, relevance; needs the Granite base model |

\*Average balanced accuracy, from the [leaderboard](https://llm-aggrefact.github.io/) and papers. The top score has barely moved since 2024.

**Verdict:** sentence-level support checking at ≤1B under a permissive license is **solved well enough**. A new small model would struggle to beat FactCG or MiniCheck on this alone.

## 4. Should this query retrieve at all?

- **No maintained drop-in gate found:** there is no open, maintained "should I retrieve?" classifier that works with API LLMs.
  - The methods with good numbers need logprobs ([FLARE](https://arxiv.org/abs/2305.06983)), hidden states ([UAR](https://arxiv.org/html/2406.12534v2), [SeaKR](https://arxiv.org/abs/2406.19215), [DRAGIN](https://github.com/oneal2000/DRAGIN)), or RL-training the generator ([Search-R1](https://github.com/PeterGriffinJin/Search-R1)).
- **The savings are small:**
  - A 35-method comparison ([ACL 2025](https://aclanthology.org/2025.acl-long.319/)) found simple uncertainty baselines as good as dedicated methods. The best skips about 19% of retrievals but nearly doubles LLM calls.
  - A TF-IDF + SVM router saves 28% of tokens ([RAGRouter-Bench](https://arxiv.org/abs/2604.03455)).
- **Real traffic points the other way:** [one production study](https://arxiv.org/abs/2605.27220) found that deciding *before* retrieval failed ("the need... is only revealed after searching the index"). Deciding *after* retrieval worked: 72% of queries took the cheap path, latency fell 32%, and quality went up.

**Verdict:** a pre-retrieval gate is weak value. Judge the context **after** retrieving it.

## 5. Licensing (not legal advice)

- **The inherited labels are risky:**
  - `selfrag_train_data` is labeled MIT, but its reflection tokens are outputs of a **Llama 2** critic trained on **GPT-4** labels.
  - The Llama 2 license says: "You will not use … any output or results of the Llama Materials to improve any other large language model" ([license](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama2/LICENSE)).
  - OpenAI's terms forbid using output "to develop models that compete" ([services agreement](https://openai.com/policies/services-agreement/)). That binds the party that called the API; no ruling on downstream users was found.
- **44% of the rows come from non-commercial sources:** 64,727 of 145,619 are gpt4_alpaca (26,168), stanford_alpaca (25,153) or sharegpt (13,406). Their datasets are CC BY-NC or come from scraped ChatGPT text ([Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)). Counted from the `dataset_name` field. Dropping them leaves 80,892 rows (FLAN, OASST1, WoW, NQ, FEVER, ASQA, OBQA, ARC-Easy); some of those are CC BY-SA.
- **Clean path:**
  - Keep the Self-RAG *prompts, schema and passages*, which are MIT code.
  - Drop the non-commercial rows above.
  - **Relabel** with an Apache-2.0 teacher such as Qwen3-32B or gpt-oss. Not Qwen2.5-3B/72B, which use the "qwen" license with a "Built with Qwen" requirement.
  - Train an Apache/MIT student (Qwen3-0.6B/1.7B, SmolLM2, ModernBERT or DeBERTa-v3).
  - Use the Self-RAG labels only to check agreement internally.
  - Avoid Gemma (distillation creates a "Model Derivative") and Llama 3.2 (naming and attribution terms) as students.

## So what is worth building?

1. **Now: explain the Self-RAG reproduction gap.** Run Phase 1 (free on Kaggle) and publish which factor accounts for which points. That answers open issues #57 and #71 and the FlashRAG discrepancy, with a test-backed harness.
2. **Next: a post-retrieval critic, only where the gap is real.** One permissively licensed ≤2B model that, in a single pass, judges **context sufficiency** ("can this passage answer the question?") and **per-sentence support**, with:
   - **calibrated probabilities:** no source reported calibration (ECE);
   - **published CPU latency:** only HHEM reports any;
   - **evaluation beyond Wikipedia QA:** the gating results above all come from Wikipedia-style benchmarks.

   **Bar to clear:** match FactCG-DeBERTa-L (75.6) on support while adding sufficiency. If it can't, contribute calibration and benchmarks to the existing tools instead.
3. **Drop:** the pre-retrieval "should I retrieve?" gate, and shipping a model trained directly on the inherited Self-RAG labels.
