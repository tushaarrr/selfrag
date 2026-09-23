"""Self-RAG short-form eval (PLAN-revised Phase 1, reused in Phase 4).

Mirrors reference/self-rag/retrieval_lm/run_short_form.py @ 1fcdc42, which crashes as shipped (PLAN-revised #4).
  generate  runs every branch the reference could take, for every item, on vLLM 0.2.6. One JSONL line per item.
  score     replays the reference's retrieve decision, candidate scoring and aggregation offline.
Generating both branches means any threshold formula can be scored without another GPU pass.
"""
import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REF_SHA = "1fcdc42"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "eval_data"

TASKS = {  # PLAN-revised Phase 1 table; flags from the reference README
    "popqa": dict(file="popqa_longtail_w_gs.jsonl", ndocs=10, max_new_tokens=100, task=None, target=54.9),
    "triviaqa": dict(file="triviaqa_test_w_gs.jsonl", ndocs=10, max_new_tokens=100, task=None, target=66.4),
    "triviaqa_plain": dict(file="triviaqa_test.jsonl", ndocs=5, max_new_tokens=100, task=None, target=66.4),  # stop-rule fallback
    "pubhealth": dict(file="health_claims_processed.jsonl", ndocs=5, max_new_tokens=50, task="fever", target=72.4),
    "arc": dict(file="arc_challenge_processed.jsonl", ndocs=5, max_new_tokens=50, task="arc_c", target=67.3),
}

# ids from data/hf/selfrag_llama2_7b/added_tokens.json, asserted against the loaded tokenizer
TOK = {"[No Retrieval]": 32000, "[Retrieval]": 32001, "[Continue to Use Evidence]": 32002,
       "[Irrelevant]": 32003, "[Relevant]": 32004, "<paragraph>": 32005, "</paragraph>": 32006,
       "[Utility:1]": 32007, "[Utility:2]": 32008, "[Utility:3]": 32009, "[Utility:4]": 32010, "[Utility:5]": 32011,
       "[Fully supported]": 32012, "[Partially supported]": 32013, "[No support / Contradictory]": 32014}
REL = ["[Irrelevant]", "[Relevant]"]
GRD = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UT = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
REFLECT_IDS = set(TOK.values())
# utils.py:52, order matters ("[No Retrieval]" before "[Retrieval]")
CONTROL = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
           "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]",
           "[Utility:4]", "[Utility:5]"]

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
TASK_INST = {"fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice."}
FORMULAS = ["released", "paper", "always", "never"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(path, **extra):
    """Written before any model loads, so a dead run still says what it was."""
    info = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), argv=sys.argv, host=platform.node(),
                python=platform.python_version(), reference_sha=REF_SHA, **extra)
    for mod in ("vllm", "torch", "transformers", "peft"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=1))


def build_instruction(row, task):
    """preprocess_input_data (run_short_form.py:210-249). Returns (instruction, answers)."""
    if task == "arc_c":
        # Only A-D and 1-4 are mapped, so a fifth option "E" is dropped, as in the reference.
        to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D"}
        opts = {to_letter[k]: t for k, t in zip(row["choices"]["label"], row["choices"]["text"]) if k in to_letter}
        opts.setdefault("D", "")
        choices = "\nA: {A}\nB: {B}\nC: {C}\nD: {D}".format(**opts)
        return TASK_INST[task] + "\n\n### Input:\n" + row["question"] + choices, [row["answerKey"]]
    q = row["question"]
    return (TASK_INST[task] + "\n\n## Input:\n\n" + q if task in TASK_INST else q), row["answers"]


def load_items(task_name):
    cfg = TASKS[task_name]
    items = []
    for line in open(DATA / cfg["file"]):
        row = json.loads(line)
        instruction, answers = build_instruction(row, cfg["task"])
        items.append(dict(prompt=PROMPT.format(instruction=instruction), answers=answers,
                          ctxs=row["ctxs"][:cfg["ndocs"]]))
    return items


def evidence_prompt(prompt, ctx):
    return prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(ctx["title"], ctx["text"])


def load_jsonl_by_idx(path):
    recs = {}
    for line in open(path):
        if not line.endswith("\n"):  # a write cut off by a kill; generate() trims it before appending
            break
        rec = json.loads(line)
        if rec["idx"] in recs:
            raise ValueError(f"duplicate idx {rec['idx']} in {path}")
        recs[rec["idx"]] = rec
    return dict(sorted(recs.items()))


# ---------------------------------------------------------------- generate (GPU)

def candidate(o):
    """What scoring reads from one retrieval-branch output: reflection-token logprobs at position 0 and at every
    position that emitted a reflection token. A token missing from vLLM's top-5000 dict stays missing."""
    lps = {pos: {t: d[t] for t in REFLECT_IDS if t in d}
           for pos, (tid, d) in enumerate(zip(o.token_ids, o.logprobs)) if pos == 0 or tid in REFLECT_IDS}
    return {"text": o.text, "ids": list(o.token_ids), "cum_lp": o.cumulative_logprob, "lps": lps}


def generate(llm, items, max_new_tokens, out_path, chunk=32, limit=None):
    from vllm import SamplingParams
    if Path(out_path).exists():
        text = Path(out_path).read_bytes()
        Path(out_path).write_bytes(text[:text.rfind(b"\n") + 1])  # drop a line cut off by a kill
    done = set(load_jsonl_by_idx(out_path)) if Path(out_path).exists() else set()
    todo = [i for i in range(len(items) if limit is None else min(limit, len(items))) if i not in done]
    greedy = dict(temperature=0.0, top_p=1.0)
    t0, n_tok = time.time(), 0
    with open(out_path, "a") as f:
        for s in range(0, len(todo), chunk):
            idxs = todo[s:s + chunk]
            prompts = [items[i]["prompt"] for i in idxs]
            # The reference's first call (logprobs=32016, run_short_form.py:57-63) only ever uses logprobs[0]: its
            # text never reaches the answer. One token gives the same distribution without 100 full-vocab dicts.
            first = llm.generate(prompts, SamplingParams(**greedy, max_tokens=1, logprobs=32016))
            noret = llm.generate([p + "[No Retrieval]" for p in prompts],
                                 SamplingParams(**greedy, max_tokens=max_new_tokens))
            ret = iter(llm.generate([evidence_prompt(items[i]["prompt"], c) for i in idxs for c in items[i]["ctxs"]],
                                    SamplingParams(**greedy, max_tokens=max_new_tokens, logprobs=5000)))
            lines = []
            for i, fo, no in zip(idxs, first, noret):
                lp0 = fo.outputs[0].logprobs[0]
                cands = [candidate(next(ret).outputs[0]) for _ in items[i]["ctxs"]]
                n_tok += len(no.outputs[0].token_ids) + sum(len(c["ids"]) for c in cands)
                lines.append(json.dumps({"idx": i,
                                    "ret_lp": {t: lp0[TOK[t]] for t in ("[Retrieval]", "[No Retrieval]",
                                                                         "[Continue to Use Evidence]")},
                                    "top10": sorted(lp0.items(), key=lambda kv: -kv[1])[:10],
                                    "noret_text": no.outputs[0].text,
                                    "cands": cands}) + "\n")
            f.write("".join(lines))  # one write per chunk
            f.flush()
            del first, noret, ret  # free this chunk's logprob dicts before the next generate call
            el = time.time() - t0
            n = s + len(idxs)
            print(f"{n}/{len(todo)} items  {el:.0f}s  {n_tok / el:.0f} out tok/s  "
                  f"{el / n:.2f}s/item -> {el / n * len(items) / 3600:.2f}h for all {len(items)}", flush=True)


# ---------------------------------------------------------------- score (CPU)

def postprocess(answer):
    """postprocess_answer_option_conditioned (run_short_form.py:36-48)."""
    for token in CONTROL:
        answer = answer.replace(token, "")
    return answer.replace("</s>", "").replace("\n", "").replace("<|endoftext|>", "")


def retrieve_score(ret_lp, formula):
    lr, ln = ret_lp["[Retrieval]"], ret_lp["[No Retrieval]"]
    if formula == "released":  # run_short_form.py:80: a ratio of log-probs
        return lr / (lr + ln)
    return math.exp(lr) / (math.exp(lr) + math.exp(ln))  # paper A.3; run_long_form_static.py:186-189


def score_cand(c, w_rel, w_sup, w_use):
    """run_short_form.py:96-157 with --use_seqscore; same operation order, so float ties break the same way."""
    lps = {int(p): {int(t): v for t, v in d.items()} for p, d in c["lps"].items()}

    def probs(pos, names):  # a token outside the top-5000 counts as logprob -100
        return {n: np.exp(float(lps[pos].get(TOK[n], -100))) for n in names}

    def first(names):
        return next((i for i, t in enumerate(c["ids"]) if t in {TOK[n] for n in names}), None)

    rel = probs(0, REL)
    relevance = rel["[Relevant]"] / (np.sum(list(rel.values())))
    ground = utility = 0.0
    if (g := first(GRD)) is not None:
        p = probs(g, GRD)
        s = np.sum(list(p.values()))
        ground = (p["[Fully supported]"] / s) + 0.5 * (p["[Partially supported]"] / s)
    if (u := first(UT)) is not None:
        p = probs(u, UT)
        s = np.sum(list(p.values()))
        utility = np.sum([w * (p[n] / s) for w, n in zip([-1, -0.5, 0, 0.5, 1], UT)])
    seq = c["cum_lp"] / max(len(c["ids"]), 1)
    final = np.exp(seq) + w_rel * relevance + w_sup * ground + w_use * utility
    return dict(final=float(final), seq=float(np.exp(seq)), relevance=float(relevance), ground=float(ground),
                utility=float(utility))


def strict(pred, answers, task):
    """Secondary metric: the answer must lead with the letter / label, not merely contain it."""
    if task == "arc_c":
        gold = {"1": "A", "2": "B", "3": "C", "4": "D"}.get(answers[0], answers[0])
        m = re.match(r"\s*([A-E])\b", pred)
    elif task == "fever":
        gold = answers[0]
        m = re.match(r"\s*(true|false)\b", pred, re.I)
    else:
        return None
    return int(bool(m) and m.group(1).lower() == gold.lower())


def score_item(item, rec, formula, threshold=0.2, w_rel=1.0, w_sup=1.0, w_use=1.0, task=None):
    s = {f: retrieve_score(rec["ret_lp"], f) for f in ("released", "paper")}
    do_ret = {"released": s["released"] > threshold, "paper": s["paper"] > threshold,
              "always": True, "never": False}[formula]
    cands = []
    if not do_ret:
        pred = postprocess(rec["noret_text"])
    else:
        cands = [score_cand(c, w_rel, w_sup, w_use) for c in rec["cands"]]
        if task in ("fever", "arc_c"):  # closed: sum scores per postprocessed answer
            answer2score = {}
            for c, sc in zip(rec["cands"], cands):
                a = postprocess(c["text"])
                answer2score[a] = answer2score.get(a, 0) + sc["final"]
            pred = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:  # open: best single path, raw text
            best = sorted(enumerate(sc["final"] for sc in cands), key=lambda x: x[1], reverse=True)[0][0]
            pred = rec["cands"][best]["text"]
    # run_short_form.py:331 raises IndexError on an empty pred; we keep going and flag it
    if pred and pred[0] in "#:":
        pred = pred[1:]
    m = "true" if "SUPPORTS" in pred else "false" if "REFUTES" in pred else pred
    return dict(idx=rec["idx"], retrieve_score=s, do_retrieve=do_ret, top10=rec["top10"], cands=cands, pred=pred,
                match=int(any(gt in m for gt in item["answers"])), strict=strict(m, item["answers"], task),
                empty_pred=not pred)


def score(task_name, run_dir, threshold, w_rel, w_sup, w_use):
    cfg = TASKS[task_name]
    items = load_items(task_name)
    recs = load_jsonl_by_idx(Path(run_dir) / f"{task_name}.gen.jsonl")
    rows = []
    for formula in FORMULAS:
        scored = [score_item(items[i], r, formula, threshold, w_rel, w_sup, w_use, cfg["task"]) for i, r in recs.items()]
        with open(Path(run_dir) / f"{task_name}.{formula}.scored.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in scored)
        acc = 100 * np.mean([x["match"] for x in scored])
        st = [x["strict"] for x in scored if x["strict"] is not None]
        rows.append(dict(task=task_name, formula=formula, n=len(scored), complete=len(scored) == len(items),
                         retrieval_rate=round(100 * np.mean([x["do_retrieve"] for x in scored]), 1),
                         match=round(acc, 1), strict=round(100 * np.mean(st), 1) if st else None,
                         target=cfg["target"], diff=round(acc - cfg["target"], 1),
                         empty_preds=sum(x["empty_pred"] for x in scored),
                         threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use))
    with open(Path(run_dir) / "summary.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    for r in rows:
        flag = "  <-- >2 pts off" if r["formula"] == "released" and abs(r["diff"]) > 2 else ""
        print(" ".join(f"{k}={v}" for k, v in r.items() if k not in ("w_rel", "w_sup")) + flag)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/released")
    ap.add_argument("--model", default="selfrag/selfrag_llama2_7b")
    ap.add_argument("--limit", type=int, help="first N items only (pilot)")
    ap.add_argument("--chunk", type=int, default=32, help="items per vLLM call; bounded by host RAM for logprob dicts")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs, e.g. 2 on Kaggle's 2x T4")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--w_rel", type=float, default=1.0)
    ap.add_argument("--w_sup", type=float, default=1.0)
    ap.add_argument("--w_use", type=float, default=1.0, help="1.0 as released (argparse default); the paper says 0.5")
    a = ap.parse_args()
    cfg, out = TASKS[a.task], Path(a.out)
    if a.cmd == "score":
        return score(a.task, out, a.threshold, a.w_rel, a.w_sup, a.w_use)

    items = load_items(a.task)
    n = len(items) if a.limit is None else min(a.limit, len(items))
    # Host-RAM limits (Kaggle has 30GB): TP workers load the 10GB .bin shard one at a time, no pinned swap (greedy
    # n=1 preempts by recompute, never swaps), and no CUDA graphs (host-memory leak with TP>1 in 0.2.6).
    engine = dict(dtype="half", tensor_parallel_size=a.tp, max_parallel_loading_workers=1, swap_space=0,
                  enforce_eager=True)
    write_manifest(out / f"{a.task}.manifest.json", model=a.model, task=a.task, cfg=cfg, limit=a.limit,
                   chunk=a.chunk, engine=engine, input_sha256=sha256(DATA / cfg["file"]),
                   upper_bound_out_tokens=n * (cfg["ndocs"] + 1) * cfg["max_new_tokens"] + n)
    from vllm import LLM
    llm = LLM(model=a.model, **engine)
    tok = llm.get_tokenizer()
    assert all(tok.convert_tokens_to_ids(t) == i for t, i in TOK.items()), "reflection token ids differ"
    generate(llm, items, cfg["max_new_tokens"], out / f"{a.task}.gen.jsonl", a.chunk, a.limit)


if __name__ == "__main__":
    main()
