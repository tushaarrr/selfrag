"""Phase 2: rebuild Self-RAG critic inputs from train.jsonl and render the original labeling prompts.

train.jsonl carries every instruction, passage and output inline, with the Self-RAG critic's labels as tokens.
We split each output into segments, rebuild the four critic input types (Retrieve, IsRel, IsSup, IsUse) and
render them with the reference GPT-4 prompts, so an open-source teacher can relabel them.

  python critic_data.py pilot --out runs/phase2/pilot.jsonl    # 2,000 examples, stratified by gold class

Row grammar (reference/self-rag/data_creation/generator/postprocess_data.py:223,333,336,341,344,348):
  row  := lead? seg+ [Utility:k]
  seg  := [Retrieval]<paragraph>P</paragraph>([Relevant] T IsSup | [Irrelevant] T)
        | [No Retrieval] T | [Continue to Use Evidence] T
Untagged text (spaCy sentences <30 chars, postprocess_data.py:243-245) can sit before the first token (lead) or
after an IsSup token (tail); it is output text but carries no label.
"""
import argparse
import ast
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data" / "hf" / "selfrag_train_data" / "train.jsonl"
PROMPTS = ROOT / "reference" / "self-rag" / "data_creation" / "critic" / "gpt4_reward"

RET = ("[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]")
REL = ("[Relevant]", "[Irrelevant]")
SUP = ("[Fully supported]", "[Partially supported]", "[No support / Contradictory]")
UTIL = tuple(f"[Utility:{i}]" for i in range(1, 6))
PIECE = re.compile(r"(<paragraph>.*?</paragraph>|\[(?:No Retrieval|Retrieval|Continue to Use Evidence|"
                   r"Relevant|Irrelevant|Fully supported|Partially supported|"
                   r"No support / Contradictory|Utility:\d)\])", re.S)
REL_CONTAMINATED = {"nq", "fever", "wow"}  # postprocess_data.py:309-310,325-326 rewrite their IsRel labels
PLAN = {  # type -> {gold label: examples}; equal per class so rare classes are covered
    "Retrieve_initial": {"[Retrieval]": 125, "[No Retrieval]": 125},
    "Retrieve_segment": {"[Retrieval]": 84, "[No Retrieval]": 83, "[Continue to Use Evidence]": 83},
    "IsRel": {"[Relevant]": 250, "[Irrelevant]": 250},
    "IsSup": {"[Fully supported]": 167, "[Partially supported]": 167, "[No support / Contradictory]": 166},
    "IsUse": {f"[Utility:{k}]": 100 for k in range(1, 6)},
}


def parse(output):
    """-> dict(lead, segs=[{retrieve, para, isrel, text, issup, tail}], utility, flags)."""
    lead, segs, utility, flags, cur = "", [], None, [], None
    for piece in PIECE.split(output):
        if not piece:
            continue
        if utility is not None:
            flags.append("after_utility")
        if piece in RET:
            cur = dict(retrieve=piece, para=None, isrel=None, text="", issup=None, tail="")
            segs.append(cur)
        elif piece.startswith("<paragraph>"):
            if cur is None or cur["retrieve"] != "[Retrieval]" or cur["para"] is not None or cur["text"]:
                flags.append("stray_paragraph")
            else:
                cur["para"] = piece[len("<paragraph>"):-len("</paragraph>")]
        elif piece in REL:
            if cur is None or cur["para"] is None or cur["isrel"] or cur["text"]:
                flags.append("stray_isrel")
            else:
                cur["isrel"] = piece
        elif piece in SUP:
            if cur is None or cur["isrel"] != "[Relevant]" or cur["issup"]:
                flags.append("stray_issup")
            else:
                cur["issup"] = piece
        elif piece in UTIL:
            if utility is not None:
                flags.append("multi_utility")
            utility = piece
        elif piece.startswith("[Utility:"):
            flags.append("bad_utility")
        elif cur is None:
            lead += piece
        elif cur["issup"]:
            cur["tail"] += piece
        elif cur["retrieve"] == "[Retrieval]" and cur["isrel"] is None:
            flags.append("text_before_isrel")
            cur["text"] += piece
        else:
            cur["text"] += piece
    if utility is None:
        flags.append("no_utility")
    for s in segs:
        if s["retrieve"] == "[Retrieval]" and s["para"] is None:
            flags.append("retrieval_no_paragraph")
        if s["isrel"] == "[Relevant]" and s["issup"] is None:
            flags.append("relevant_no_issup")
        if not s["text"].strip():
            flags.append("empty_segment_text")
    return dict(lead=lead, segs=segs, utility=utility, flags=flags)


def preceding(p, i):
    """Text before segment i, joined like create_retrieval_data.py:89-90."""
    parts = [p["lead"]] + [x for s in p["segs"][:i] for x in (s["text"], s["tail"])]
    return " ".join(x.strip() for x in parts if x.strip())


def last_paragraph(p, i):
    """Most recent passage before segment i. train.jsonl does not store the passage the critic saw for
    segment-level Retrieve (create_retrieval_data.py:93-99), so this is a proxy."""
    for s in reversed(p["segs"][:i]):
        if s["para"] is not None:
            return s["para"]
    return None


def plain_output(output):
    """Output without passages and reflection tokens (the IsUse input). Puts back the space that
    postprocess_data.py:333-344 dropped where a token sat between two sentences."""
    out = ""
    for piece in PIECE.split(output):
        if not piece or PIECE.fullmatch(piece):
            continue
        if out and not out[-1].isspace() and not piece[0].isspace():
            out += " "
        out += piece
    return out


def prompt_dict(fname):
    """The reference PROMPT_DICT, read without importing the script (it imports openai 0.28)."""
    tree = ast.parse((PROMPTS / fname).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROMPT_DICT")


def candidates(row):
    """Yield (type, gold, seg_idx) for every example this row can contribute to the pilot pools."""
    ds, out = row["dataset_name"], row["output"]
    p = parse(out)
    segs = p["segs"]
    single_noret = len(segs) == 1 and segs[0]["retrieve"] == "[No Retrieval]" and not p["lead"].strip()
    # fever outputs are true/false and had [Utility:1] forced to 5 (postprocess_data.py:211-212)
    if p["utility"] and segs and plain_output(out).strip().lower() not in ("true", "false"):
        yield "IsUse", p["utility"], -1
    for i, s in enumerate(segs):
        initial = i == 0 and not p["lead"].strip()
        if not s["text"].strip():
            continue
        # a leading [No Retrieval] in a multi-segment row can't come from postprocess_data.py:249-250; skip it
        if initial and (s["retrieve"] == "[Retrieval]" or single_noret):
            yield "Retrieve_initial", s["retrieve"], i
        elif not initial and last_paragraph(p, i) is not None:
            yield "Retrieve_segment", s["retrieve"], i
        if s["para"] is None:
            continue
        if ds not in REL_CONTAMINATED and s["isrel"]:
            yield "IsRel", s["isrel"], i
        if s["isrel"] == "[Relevant]" and s["issup"]:
            yield "IsSup", s["issup"], i


def render(row, typ, seg_idx, prompts):
    """The example dict with its teacher prompt, rendered from the verbatim reference templates."""
    p = parse(row["output"])
    ex = {"instruction": row["instruction"]}
    if typ == "IsUse":
        ex["output"] = plain_output(row["output"])
        return ex, prompts["util"]["context"].format_map(ex)
    s = p["segs"][seg_idx]
    pre = preceding(p, seg_idx)
    ex |= {"target_output": s["text"], "preceding_sentences": pre, "sent_idx": seg_idx}
    if typ == "Retrieve_initial":  # the teacher answers [Yes]/[No]
        return ex, prompts["ret"]["context"].format_map(ex)
    if typ == "Retrieve_segment":
        ex["evidence"] = last_paragraph(p, seg_idx)
        # chatgpt_need_retrieval.py:132-141 swaps these two keys; use the template that matches the input
        key = "multi_retrieval_three_way" + ("" if pre else "_no_preceding")
        return ex, prompts["ret"][key].format_map(ex)
    ex["evidence"] = s["para"]
    key = "multi" if pre else "multi_no_preceding"  # chatgpt_relevance.py:95-98, chatgpt_groundness.py:116-119
    return ex, prompts["rel" if typ == "IsRel" else "sup"][key].format_map(ex)


def build_pilot(train_path, out_path, seed=0):
    """Two passes over train.jsonl: pools per (type, gold), then stratified sampling and prompt rendering.
    weight = pool size / sampled, to reweight per-class results back to the natural class mix."""
    pools = defaultdict(list)
    with open(train_path) as f:
        for ln, line in enumerate(f):
            for typ, gold, si in candidates(json.loads(line)):
                pools[(typ, gold)].append((ln, si))
    rng = random.Random(seed)
    chosen = defaultdict(list)  # line -> [(seg_idx, type, gold, weight)]
    for typ, alloc in PLAN.items():
        used = set()  # at most one example per row per type
        for gold, n in alloc.items():
            pool = pools[(typ, gold)][:]
            rng.shuffle(pool)
            picked = []
            for ln, si in pool:
                if ln not in used:
                    used.add(ln)
                    picked.append((ln, si))
                if len(picked) == n:
                    break
            for ln, si in picked:
                chosen[ln].append((si, typ, gold, len(pools[(typ, gold)]) / max(1, len(picked))))
    prompts = {k: prompt_dict(f) for k, f in (("ret", "chatgpt_need_retrieval.py"), ("rel", "chatgpt_relevance.py"),
                                              ("sup", "chatgpt_groundness.py"), ("util", "chatgpt_utility.py"))}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_path) as f, open(out_path, "w") as w:
        for ln, line in enumerate(f):
            if ln not in chosen:
                continue
            row = json.loads(line)
            for si, typ, gold, wt in chosen[ln]:
                ex, prompt = render(row, typ, si, prompts)
                w.write(json.dumps({"id": row["id"], "line": ln, "seg_idx": si, "dataset_name": row["dataset_name"],
                                    "type": typ, "gold": gold, "weight": round(wt, 3), **ex, "prompt": prompt}) + "\n")
    return {f"{t} {g}": len(v) for (t, g), v in sorted(pools.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot"])
    ap.add_argument("--out", default="runs/phase2/pilot.jsonl")
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k, v in build_pilot(a.train, a.out, a.seed).items():
        print(f"pool {k}: {v}")
    print("md5", hashlib.md5(Path(a.out).read_bytes()).hexdigest(), a.out)


if __name__ == "__main__":
    main()
