"""Oracle test: the reference run_short_form.main() and our generate+score must agree item by item.

Both run against the same fake vLLM on real eval rows. The reference is exec'd read-only with one patch: the
max_depth kwarg that crashes it (PLAN-revised #4). No GPU, no checkpoint download.
"""
import hashlib
import json
import random
import re
import sys
import types
import zlib
from pathlib import Path

import pytest

import eval_selfrag as ev

REF = ev.ROOT / "reference" / "self-rag" / "retrieval_lm"
TOKENIZER = ev.ROOT / "data" / "hf" / "selfrag_llama2_7b"
GRD_IDS = [ev.TOK[t] for t in ev.GRD]
UT_IDS = [ev.TOK[t] for t in ev.UT]
N = 40


class SamplingParams:
    def __init__(self, temperature=1.0, top_p=1.0, max_tokens=16, logprobs=None):
        self.max_tokens, self.logprobs = max_tokens, logprobs


class FakeLLM:
    """Deterministic per prompt, like greedy decoding: the sequence depends on the prompt only and is truncated to
    max_tokens. Reflection ids fall out of the top-k dict unless logprobs covers the whole vocab."""

    def __init__(self, *a, **k):
        pass

    def generate(self, prompts, sp):
        return [self._one(p, sp) for p in prompts]

    def _one(self, p, sp):
        seed = hashlib.md5(p.encode()).hexdigest()
        rng = random.Random(seed)
        words = re.findall(r"\S+", p)[-60:] + ["A", "B", "C", "D", "true", "false", "SUPPORTS", "#", ":"]
        seq = []
        if p.endswith("</paragraph>"):
            seq.append(rng.choice([ev.TOK["[Relevant]"], ev.TOK["[Irrelevant]"]]))
        elif p.endswith("### Response:\n"):
            seq.append(rng.choice([ev.TOK["[Retrieval]"], ev.TOK["[No Retrieval]"]]))
        seq += [rng.choice(words) for _ in range(rng.randint(1, 4))]
        for ids, prob in ((GRD_IDS, 0.7), (UT_IDS, 0.7), (UT_IDS, 0.3)):
            if rng.random() < prob:
                seq.append(rng.choice(ids))
        seq = (seq + [2])[:sp.max_tokens]  # EOS is kept in token_ids, as in vLLM 0.2.6
        token_ids, logprobs, cum = [], [], 0.0
        for pos, t in enumerate(seq):
            tid = t if isinstance(t, int) else 3 + zlib.crc32(t.encode()) % 31990
            prng = random.Random(f"{seed}/{pos}")
            d = {}
            for rid in sorted(ev.REFLECT_IDS):
                lp, keep = -prng.uniform(0.6, 12), prng.random() < 0.6
                if (sp.logprobs or 0) >= 32016 or keep:
                    d[rid] = lp
            d[tid] = -prng.uniform(0, 0.5)
            token_ids.append(tid)
            logprobs.append(d)
            cum += d[tid]
        text = " ".join(t for t in seq if isinstance(t, str))  # skip_special_tokens=True
        out = types.SimpleNamespace(text=text, token_ids=token_ids, cumulative_logprob=cum,
                                    logprobs=logprobs if sp.logprobs else None)
        return types.SimpleNamespace(outputs=[out])


@pytest.fixture(scope="module", autouse=True)
def ref():
    """Fake vllm/spacy/openai for this module only, plus the exec'd reference namespace."""
    import transformers  # noqa: F401  (before the stubs: it runs find_spec on "openai")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=SamplingParams))
        mp.setitem(sys.modules, "spacy", types.ModuleType("spacy"))
        mp.setitem(sys.modules, "openai", types.ModuleType("openai"))
        mp.syspath_prepend(str(REF))
        ns = {"__name__": "reference"}
        exec(compile((REF / "run_short_form.py").read_text(), "run_short_form.py", "exec"), ns)
        orig = ns["call_model_rerank_w_scores_batch"]
        ns["call_model_rerank_w_scores_batch"] = lambda *a, max_depth=None, **k: orig(*a, **k)
        yield ns
    for name in ("utils", "metrics"):  # the reference's own modules
        sys.modules.pop(name, None)


def run_reference(ns, task_name, tmp_path, mode):
    cfg = ev.TASKS[task_name]
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(open(ev.DATA / cfg["file"]).readlines()[:N]))
    argv = ["--model_name", str(TOKENIZER), "--input_file", str(inp), "--output_file", str(tmp_path / "ref.json"),
            "--max_new_tokens", str(cfg["max_new_tokens"]), "--threshold", "0.2", "--metric", "match",
            "--ndocs", str(cfg["ndocs"]), "--use_groundness", "--use_utility", "--use_seqscore", "--dtype", "half"]
    argv += ["--task", cfg["task"]] if cfg["task"] else []
    argv += ["--mode", mode] if mode else []
    sys.argv = ["run_short_form.py", *argv]
    ns["main"]()
    return json.loads((tmp_path / "ref.json").read_text())


@pytest.mark.parametrize("task_name,mode,formula", [
    ("popqa", "adaptive_retrieval", "released"), ("triviaqa", "adaptive_retrieval", "released"),
    ("pubhealth", None, "released"), ("arc", None, "released"),
    ("popqa", "always_retrieve", "always"), ("arc", "no_retrieval", "never"),
])
def test_matches_reference(ref, tmp_path, task_name, mode, formula):
    want = run_reference(ref, task_name, tmp_path, mode)
    cfg = ev.TASKS[task_name]
    items = ev.load_items(task_name)[:N]
    ev.generate(FakeLLM(), items, cfg["max_new_tokens"], tmp_path / "gen.jsonl", chunk=7)
    recs = ev.load_jsonl_by_idx(tmp_path / "gen.jsonl")
    got = [ev.score_item(items[i], r, formula, task=cfg["task"]) for i, r in recs.items()]
    assert [g["idx"] for g in got] == list(range(N))
    retrieved = 0
    for g, pred, m, res in zip(got, want["preds"], want["metric_results"], want["all_results"]):
        paths = [res[k]["score"] for k in res if k.startswith("retrieval_")]
        retrieved += g["do_retrieve"]
        assert g["do_retrieve"] == bool(paths)
        assert [c["final"] for c in g["cands"]] == paths
        assert (g["pred"], g["match"]) == (pred, m)
    if formula == "released":  # both branches exercised
        assert 0 < retrieved < N


def test_prompts_match_reference_on_every_item(ref):
    for task_name, cfg in ev.TASKS.items():
        rows = ref["preprocess_input_data"]([json.loads(l) for l in open(ev.DATA / cfg["file"])], task=cfg["task"])
        for ours, row in zip(ev.load_items(task_name), rows, strict=True):
            prompt, evidences = ref["process_data_evidences"](row, top_n=cfg["ndocs"])
            assert (ours["prompt"], ours["ctxs"], ours["answers"]) == (prompt, evidences, row["answers"])


def test_token_ids_match_tokenizer(ref):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    assert all(tok.convert_tokens_to_ids(t) == i for t, i in ev.TOK.items())


def test_retrieve_formulas_run_opposite():
    confident = {"[Retrieval]": -0.01, "[No Retrieval]": -4.6}  # p(Ret) ~ 0.99
    assert ev.retrieve_score(confident, "paper") > 0.2 > ev.retrieve_score(confident, "released")
    unlikely = {"[Retrieval]": -4.6, "[No Retrieval]": -0.01}  # p(Ret) ~ 0.01
    assert ev.retrieve_score(unlikely, "released") > 0.2 > ev.retrieve_score(unlikely, "paper")


def test_strict_metric():
    assert ev.strict(" C", ["C"], "arc_c") == 1
    assert ev.strict("Carbon", ["C"], "arc_c") == 0  # match() would score this 1
    assert ev.strict("B: text", ["2"], "arc_c") == 1  # numeric gold keys map to letters
    assert ev.strict("false.", ["false"], "fever") == 1
    assert ev.strict("it is true", ["true"], "fever") == 0


def test_resume_and_duplicates(tmp_path):
    items = ev.load_items("arc")[:10]
    out = tmp_path / "gen.jsonl"
    ev.generate(FakeLLM(), items, 50, out, chunk=4, limit=6)
    ev.generate(FakeLLM(), items, 50, out, chunk=4)
    assert list(ev.load_jsonl_by_idx(out)) == list(range(10))
    full = out.read_text()
    out.write_text(full + full[:50])  # killed mid-write
    assert list(ev.load_jsonl_by_idx(out)) == list(range(10))
    ev.generate(FakeLLM(), items, 50, out, chunk=4)
    assert out.read_text() == full
    out.write_text(full + full.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_jsonl_by_idx(out)
