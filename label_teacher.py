"""Phase 2: relabel critic examples with an open-source teacher and measure agreement with Self-RAG's labels.

  label   runs the teacher (vLLM) over critic_data.py's prompts. One JSONL line per example, resumable.
  report  per type: agreement with the inline Self-RAG label, per-class recall, confusion matrix, Cohen's kappa.

The gold labels are predictions of Self-RAG's Llama-2-7B critic, which agreed with GPT-4 on 93.8 / 80.2 / 93.5 /
73.5% of Retrieve / IsRel / IsSup / IsUse (paper appendix), so agreement is capped by that noise.
"""
import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from critic_data import REL, RET, SUP
from eval_selfrag import load_jsonl_by_idx, sha256, write_manifest

NONCOMMERCIAL = {"gpt4_alpaca", "stanford_alpaca", "sharegpt"}  # LANDSCAPE.md section 5
OPTIONS = {"Retrieve_segment": RET, "IsRel": REL, "IsSup": SUP}


def parse_label(typ, text):
    """The teacher's label, from the part before 'Explanation:' as in the reference postprocess functions.
    Returns the Self-RAG token, or None if no label is found."""
    head = text.split("Explanation:")[0]
    if typ == "IsUse":
        m = re.search(r"[1-5]", head)
        return f"[Utility:{m.group()}]" if m else None
    if typ == "Retrieve_initial":  # the prompt asks for [Yes]/[No]; combine_chat_gpt_reward.py:141-146
        m = re.search(r"\b(yes|no)\b", head, re.I)
        return {"yes": "[Retrieval]", "no": "[No Retrieval]"}[m.group(1).lower()] if m else None
    norm = re.sub(r"\s+", " ", head).lower()
    # earliest mention wins; for a tie at one position, the longer option ("[No Retrieval]" over "[Retrieval]")
    hits = [(norm.find(o.lower()), -len(o), o) for o in OPTIONS[typ] if o.lower() in norm]
    return min(hits)[2] if hits else None


def label(llm, examples, out_path, max_tokens=160, max_model_len=4096, chunk=256):
    """Greedy, non-thinking chat completions. Examples go in template order so prefix caching reuses the
    few-shot block. Prompts too long for the context window are recorded, not dropped."""
    from vllm import SamplingParams
    if Path(out_path).exists():
        text = Path(out_path).read_bytes()
        Path(out_path).write_bytes(text[:text.rfind(b"\n") + 1])  # drop a line cut off by a kill
    done = set(load_jsonl_by_idx(out_path)) if Path(out_path).exists() else set()
    tok = llm.get_tokenizer()
    todo = sorted((i for i in range(len(examples)) if i not in done), key=lambda i: examples[i]["type"])
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0, n_in, n_out = time.time(), 0, 0
    with open(out_path, "a") as f:
        for s in range(0, len(todo), chunk):
            idxs = todo[s:s + chunk]
            chats = [tok.apply_chat_template([{"role": "user", "content": examples[i]["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for i in idxs]
            fits = [len(tok(c).input_ids) + max_tokens <= max_model_len for c in chats]
            outs = iter(llm.generate([c for c, ok in zip(chats, fits) if ok], sp))
            lines = []
            for i, ok in zip(idxs, fits):
                ex = examples[i]
                rec = {"idx": i, "type": ex["type"], "gold": ex["gold"], "weight": ex["weight"],
                       "dataset_name": ex["dataset_name"]}
                if ok:
                    o = next(outs)
                    text = o.outputs[0].text
                    n_in, n_out = n_in + len(o.prompt_token_ids), n_out + len(o.outputs[0].token_ids)
                    rec |= {"text": text, "label": parse_label(ex["type"], text)}
                else:
                    rec |= {"text": None, "label": None, "too_long": True}
                lines.append(json.dumps(rec) + "\n")
            f.write("".join(lines))
            f.flush()
            el = time.time() - t0
            n = s + len(idxs)
            print(f"{n}/{len(todo)}  {el:.0f}s  {n_in / el:.0f} prompt tok/s  {n_out / el:.0f} out tok/s  "
                  f"{el / n:.3f}s/example -> {el / n * 30000 / 3600:.1f}h per 30k", flush=True)


def kappa(pairs):
    """Cohen's kappa for (gold, pred) pairs; pred may be None (a parse failure counts as its own class)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(g == p for g, p in pairs) / n
    cg, cp = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
    pe = sum(cg[k] * cp[k] for k in cg) / n / n
    return None if pe == 1 else (po - pe) / (1 - pe)


def report(labels_path):
    by_type = defaultdict(list)
    for r in load_jsonl_by_idx(labels_path).values():
        by_type[r["type"]].append(r)
    rows = {}
    for typ, rs in sorted(by_type.items()):
        scored = [r for r in rs if not r.get("too_long")]
        pairs = [(r["gold"], r["label"]) for r in scored]
        w = sum(r["weight"] for r in scored)
        classes = sorted({g for g, _ in pairs})
        clean = [(r["gold"], r["label"]) for r in scored if r["dataset_name"] not in NONCOMMERCIAL]
        rows[typ] = dict(
            n=len(rs), too_long=len(rs) - len(scored), parse_fail=sum(p is None for _, p in pairs),
            agree=round(100 * sum(g == p for g, p in pairs) / max(1, len(pairs)), 1),
            agree_natural_mix=round(100 * sum(r["weight"] for r in scored if r["gold"] == r["label"]) / w, 1) if w else None,
            kappa=round(kappa(pairs), 3) if pairs else None,
            agree_without_noncommercial=round(100 * sum(g == p for g, p in clean) / max(1, len(clean)), 1),
            recall={c: round(100 * sum(p == c for g, p in pairs if g == c) / sum(g == c for g, _ in pairs), 1)
                    for c in classes},
            confusion={c: dict(Counter(str(p) for g, p in pairs if g == c)) for c in classes})
    Path(labels_path).with_suffix(".report.json").write_text(json.dumps(rows, indent=1))
    for typ, r in rows.items():
        print(f"{typ:17s} n={r['n']} agree={r['agree']} natural_mix={r['agree_natural_mix']} kappa={r['kappa']} "
              f"parse_fail={r['parse_fail']} too_long={r['too_long']} recall={r['recall']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "report"])
    ap.add_argument("--examples", default="runs/phase2/pilot.jsonl")
    ap.add_argument("--out", default="runs/phase2/qwen3-32b-awq/pilot.labels.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=160, help="keeps the explanation; the label comes first")
    a = ap.parse_args()
    if a.cmd == "report":
        return report(a.out)
    examples = [json.loads(line) for line in open(a.examples)]
    # Proven on Kaggle 2x T4 (https://www.kaggle.com/code/llkh0a/qwen3-32b-awq). max_model_len must be set:
    # the model's default of 40,960 tokens needs more KV cache than a T4 has, and vLLM refuses to start.
    engine = dict(quantization="awq", dtype="half", tensor_parallel_size=a.tp, gpu_memory_utilization=0.95,
                  enforce_eager=True, max_model_len=4096, enable_prefix_caching=True)
    write_manifest(Path(a.out).parent / "manifest.json", model=a.model, engine=engine, max_tokens=a.max_tokens,
                   examples=a.examples, examples_sha256=sha256(a.examples), n=len(examples))
    os.environ.setdefault("VLLM_USE_V1", "0")  # vLLM 0.10.0 uses V0 below compute capability 8.0 anyway
    from vllm import LLM
    label(LLM(model=a.model, **engine), examples, a.out, a.max_tokens, engine["max_model_len"])
    report(a.out)


if __name__ == "__main__":
    main()
