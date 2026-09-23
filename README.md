# Self-RAG, reproduced and audited

This is a careful, testable reproduction of [Self-RAG](https://arxiv.org/abs/2310.11511) (Asai et al., 2023). It includes an eval harness that matches the original code item by item, a single-GPU trainer, and a list of problems in the original release that change how its numbers should be read.

**Status:** everything that can be checked on a CPU is built and tested. **No GPU results yet.** The numbers below are the paper's targets, not ours.

## What's wrong in the original release

Found while building this. Each point is tested or cited in [NOTES.md](NOTES.md) and [PLAN-revised.md](PLAN-revised.md).

- **The eval script crashes as shipped.** `run_short_form.py` passes a `max_depth` argument that its own function doesn't accept.
- **The retrieval threshold isn't the paper's.** The code applies 0.2 to a ratio of *log*-probabilities, while the paper uses probabilities. The code retrieves when p(Retrieve) < 0.72; the paper's rule retrieves when p(Retrieve) > 0.2. The two agree only in between.
- **The "accuracy" metric is a substring test.** On ARC, 22 of 1,172 questions have gold answers `1`-`4` while the prompt labels the options A-D, so they can never be marked correct.
- **Training masks more than the paper describes.** In the 22.8% of training rows with two or more passages, it masks everything from the first passage to the last, including the reflection tokens in between.
- **LoRA silently breaks the new tokens.** Llama-2's output layer is separate from its input embeddings, and the repo's LoRA path leaves both untrained. So the 16 reflection tokens never learn.
- **The eval-data link is dead**; `setup_data.sh` uses a verified mirror.

## Quick start (CPU, about 2 minutes)

```bash
git clone https://github.com/tushaarrr/selfrag && cd selfrag
python3.10 -m venv .venv && . .venv/bin/activate   # 3.10 or 3.11
pip install -r requirements.txt pytest jsonlines    # Linux + CUDA. On a Mac, skip vllm: pip install torch==2.1.2 transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 numpy==1.26.4 sentencepiece protobuf pytest jsonlines
./setup_data.sh                                     # ~1.2GB of inputs, every file checked against SHA256SUMS
pytest -q                                           # 14 tests, no GPU needed
```

The tests run the **original** Self-RAG code as an oracle: our harness has to reproduce its decisions, scores and predictions exactly on real eval rows, and our trainer its tokens and labels.

## Run the eval on Kaggle (free)

The released 7B model on 2× T4, all four tasks. The time estimates below are rough; the pilot run measures the real speed.

1. On kaggle.com: **Create → New Notebook**. In the right panel, set **Accelerator = GPU T4 x2** and **Internet = On** (Internet needs a phone-verified account).
2. **Pilot** (about 20-30 minutes). Run this in a cell:
   ```
   !git -C selfrag pull -q || git clone -q https://github.com/tushaarrr/selfrag
   !LIMIT=50 TASKS=popqa bash selfrag/kaggle_phase1.sh
   ```
   The first line updates an existing copy, or clones it the first time. Re-run both lines after any fix to the repo.
   It prints seconds per item and the projected hours for the full task.
3. **Full run.** Keep the first line and change the second to `!bash selfrag/kaggle_phase1.sh`, then click **Save Version → Save & Run All (Commit)**. It runs in the background for up to 11 hours and saves `/kaggle/working` as the version's output. You can close the browser.
4. **Resume**, if it stopped on the time limit:
   1. In a new version, click **Add Input** and pick the previous version's output. Pick one that finished with `runs/released` in it, not a failed one.
   2. Run `!find /kaggle/input -maxdepth 6 -type d -path '*/selfrag/runs'`. Kaggle now mounts outputs under `/kaggle/input/notebooks/<owner>/<notebook>/`.
   3. Run `!PREV=<printed path> bash selfrag/kaggle_phase1.sh`. Finished items are skipped.
5. **Results** are in `runs/released/summary.jsonl`. Each scoring appends one row per task and retrieval policy (`released`, `paper`, `always`, `never`), with accuracy, retrieval rate and the gap to the paper. After a resume, read the last row per task and policy that has `complete: true`.

Kaggle's weekly GPU quota is about 30 hours. Training doesn't fit on T4s; see NOTES.md, "Where to run".

## Files

| File | What |
|---|---|
| `eval_selfrag.py` | `generate`: runs the retrieve and no-retrieve branches for every question on vLLM 0.2.6 (resumable). `score`: replays the original decision logic offline under four retrieval policies |
| `train_generator.py` | bf16 LoRA trainer with trainable reflection-token rows, per-class reflection accuracy, resume, merge |
| `kaggle_phase1.sh` | The Kaggle runner above |
| `setup_data.sh` | Fetches and verifies the inputs |
| `NOTES.md` | Pre-registered predictions, deviations from the original, findings, runbook |
| `PLAN-revised.md` | The phase plan, with the evidence behind each change |
| `LANDSCAPE.md` | Existing RAG critics, gating research and licensing, with sources |

## Targets (paper, Table 2, Self-RAG 7B)

| PopQA | TriviaQA | PubHealth | ARC-Challenge |
|---|---|---|---|
| 54.9 | 66.4 | 72.4 | 67.3 |

## Roadmap

1. **Phase 1:** run the released 7B on the four tasks. Does it reproduce, and under which threshold formula?
2. **Phase 3:** train a generator on 30k rows with LoRA on one GPU. How far behind the released model does it land?
3. **Explain the gap.** Independent reruns and GitHub issues don't match the paper ([FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), [#57](https://github.com/AkariAsai/self-rag/issues/57), [#71](https://github.com/AkariAsai/self-rag/issues/71)). Publish which factor accounts for which points.
4. **Then, only if it clears the bar:** a permissively licensed ≤2B critic that judges, after retrieval, whether the context is sufficient and whether each sentence is supported, with calibrated scores and CPU latency. See [LANDSCAPE.md](LANDSCAPE.md) for what already exists and why a pre-retrieval gate isn't worth building.

Contributions welcome, especially GPU runs of Phase 1. Please attach `runs/released/*.manifest.json` with any results.
