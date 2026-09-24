"""Phase 2 checks on CPU: the train.jsonl parser, the pilot sample (pinned by md5), label parsing, the labeling
loop against a fake vLLM, and the agreement report."""
import hashlib
import json
import sys
import types
from collections import Counter

import pytest

import critic_data as cd
import label_teacher as lt

PILOT_MD5 = "6ee637451d9e461d04997b667e57ee2c"  # seed 0, train.jsonl sha256 c58e3687...


def test_parse_grammar():
    o = ("Hi.[Retrieval]<paragraph>T\nx</paragraph>[Relevant]A b.[Fully supported]Ok."
         "[Continue to Use Evidence]C d.[Retrieval]<paragraph>U\ny</paragraph>[Irrelevant]E f.[Utility:4]")
    p = cd.parse(o)
    assert p["lead"] == "Hi." and p["utility"] == "[Utility:4]" and not p["flags"]
    s0, s1, s2 = p["segs"]
    assert (s0["para"], s0["isrel"], s0["text"], s0["issup"], s0["tail"]) == ("T\nx", "[Relevant]", "A b.",
                                                                              "[Fully supported]", "Ok.")
    assert s1["retrieve"] == "[Continue to Use Evidence]" and s2["isrel"] == "[Irrelevant]" and s2["issup"] is None
    assert cd.preceding(p, 2) == "Hi. A b. Ok. C d." and cd.last_paragraph(p, 1) == "T\nx"
    assert cd.last_paragraph(p, 0) is None and cd.plain_output(o) == "Hi. A b. Ok. C d. E f."
    bad = cd.parse("[Retrieval][Relevant]x[Utility:5][No Retrieval]y")
    assert {"stray_isrel", "after_utility", "retrieval_no_paragraph"} <= set(bad["flags"])


def test_every_train_row_parses():
    flags = [f for line in open(cd.TRAIN) for f in cd.parse(json.loads(line)["output"])["flags"]]
    assert flags == ["empty_segment_text"] * 3  # 145,619 rows; three segments have no text


def test_pilot_is_pinned(tmp_path):
    cd.build_pilot(cd.TRAIN, tmp_path / "pilot.jsonl")
    assert hashlib.md5((tmp_path / "pilot.jsonl").read_bytes()).hexdigest() == PILOT_MD5
    exs = [json.loads(line) for line in open(tmp_path / "pilot.jsonl")]
    assert Counter(e["type"] for e in exs) == {"Retrieve_initial": 250, "Retrieve_segment": 250, "IsRel": 500,
                                               "IsSup": 500, "IsUse": 452}
    for e in exs:
        if e["type"] == "Retrieve_segment":  # the swapped keys (chatgpt_need_retrieval.py:132-141) are fixed
            assert ("Preceding sentences: " + e["preceding_sentences"] in e["prompt"]) == bool(e["preceding_sentences"])
        if e["type"] == "IsRel":
            assert e["dataset_name"] not in cd.REL_CONTAMINATED
        if e["type"] == "IsUse":  # only rows whose answer postprocess_data.py left intact
            assert e["gold"].startswith("[Utility:") and "[No Retrieval]" not in e["prompt"]



@pytest.mark.parametrize("typ,text,want", [
    ("IsRel", "[Relevant]\nExplanation: the passage is about it.", "[Relevant]"),
    ("IsRel", "[Irrelevant]\nExplanation: not [Relevant] at all.", "[Irrelevant]"),
    ("Retrieve_segment", "[No Retrieval]\nExplanation: common sense.", "[No Retrieval]"),
    ("Retrieve_segment", "Rating: [Continue to Use Evidence]\nExplanation: x", "[Continue to Use Evidence]"),
    ("Retrieve_segment", "[Retrieval]\nExplanation: needs [No Retrieval]? no.", "[Retrieval]"),
    ("IsSup", "[No support /\nContradictory]\nExplanation: x", "[No support / Contradictory]"),
    ("IsSup", "[partially supported]", "[Partially supported]"),
    ("IsUse", "Perceived utility: 4\nExplanation: worth 5 minutes", "[Utility:4]"),
    ("Retrieve_initial", "[Yes]\nExplanation: No need to", "[Retrieval]"),
    ("Retrieve_initial", "[No]", "[No Retrieval]"),
    ("IsSup", "I cannot tell.", None),
])
def test_parse_label(typ, text, want):
    assert lt.parse_label(typ, text) == want


class FakeTok:
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "<user>" + msgs[0]["content"] + "<assistant>"

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=text.split())


class FakeLLM:
    def get_tokenizer(self):
        return FakeTok()

    def generate(self, chats, sp):
        reply = {"IsRel": "[Relevant]", "IsUse": "Perceived utility: 5"}
        return [types.SimpleNamespace(prompt_token_ids=c.split(), outputs=[types.SimpleNamespace(
            text=reply["IsRel" if "Evidence" in c else "IsUse"] + "\nExplanation: fake.", token_ids=[1, 2, 3])])
            for c in chats]


def test_label_resume_too_long_and_report(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **k: k))
    exs = [dict(type="IsRel", gold=g, weight=2.0, dataset_name=d, prompt="Evidence: e")
           for g, d in (("[Relevant]", "nq"), ("[Irrelevant]", "sharegpt"), ("[Irrelevant]", "flan_v2"))]
    exs += [dict(type="IsUse", gold="[Utility:5]", weight=1.0, dataset_name="oasst1", prompt="word " * 5000)]
    out = tmp_path / "labels.jsonl"
    lt.label(FakeLLM(), exs[:2], out, chunk=1)
    lt.label(FakeLLM(), exs, out, chunk=2)  # resumes: only idx 2 and 3 are new
    recs = lt.load_jsonl_by_idx(out)
    assert list(recs) == [0, 1, 2, 3] and recs[3]["too_long"] and recs[0]["label"] == "[Relevant]"
    rows = lt.report(out)
    assert rows["IsRel"]["agree"] == 33.3 and rows["IsRel"]["recall"] == {"[Irrelevant]": 0.0, "[Relevant]": 100.0}
    assert rows["IsRel"]["agree_without_noncommercial"] == 50.0 and rows["IsUse"]["too_long"] == 1


def test_kappa():
    assert lt.kappa([("a", "a"), ("b", "b")]) == 1.0
    assert lt.kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0.0
