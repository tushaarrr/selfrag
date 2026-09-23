"""Phase 3 checks on CPU: tokenisation/masking equals the reference's, and a tiny Llama trains, resumes and merges."""
import ast
import copy
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

import train_generator as tg

FINETUNE = tg.ROOT / "reference" / "self-rag" / "retrieval_lm" / "finetune.py"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(tg.TOKENIZER, use_fast=False)


@pytest.fixture(scope="module")
def ref_encode():
    """The reference's own functions, exec'd without its module-level imports (accelerate, datasets, ...)."""
    tree = ast.parse(FINETUNE.read_text())
    keep = [n for n in tree.body if getattr(n, "name", None) in ("_tokenize_fn", "encode_with_prompt_completion_format")
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PROMPT_DICT")]
    ns = dict(torch=torch, copy=copy, transformers=transformers, Dict=dict, print=lambda *a: None)
    exec(compile(ast.Module(keep, []), str(FINETUNE), "exec"), ns)
    return ns["encode_with_prompt_completion_format"]


def rows_with_paragraphs(n_multi, n_other):
    multi, other = [], []
    for line in open(tg.TRAIN):
        r = json.loads(line)
        (multi if r["output"].count("<paragraph>") >= 2 else other).append(r)
        if len(multi) >= n_multi and len(other) >= n_other:
            return multi[:n_multi] + other[:n_other]


def test_encode_matches_reference(tok, ref_encode):
    rows = rows_with_paragraphs(150, 150)
    for max_len in (2048, 128):  # 128 exercises truncation, including cut-off paragraphs
        for r in rows:
            want = ref_encode(r, tokenizer=tok, max_seq_length=max_len,
                              context_markups=[tg.TOK["<paragraph>"], tg.TOK["</paragraph>"]])
            got = tg.encode(r, tok, max_len)
            assert got["input_ids"] == want["input_ids"].tolist()
            assert got["labels"] == want["labels"].tolist()


def test_split_is_seeded_and_disjoint():
    a = tg.split(1000, 300, 0.01, 42)
    assert a == tg.split(1000, 300, 0.01, 42)
    assert len(a[0]) == 300 and len(a[1]) == 10 and not set(a[0]) & set(a[1])


def test_tiny_train_resume_merge(tmp_path, capsys):
    base = tmp_path / "base"
    LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, max_position_embeddings=512)).save_pretrained(base)
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows_with_paragraphs(60, 140)))
    out = tmp_path / "run"
    common = ["--out", str(out), "--base", str(base), "--train_file", str(data), "--n_train", "120",
              "--heldout_frac", "0.05", "--max_len", "256", "--micro_bs", "4", "--accum", "1", "--lr", "1e-3",
              "--save_steps", "30"]
    tg.main(common + ["--max_steps", "60"])
    log = capsys.readouterr().out
    assert "step 50:" in log and "projected" in log
    assert json.loads((out / "checkpoint-60" / "trainer_state.json").read_text())["global_step"] == 60

    report = tg.main(common + ["--max_steps", "90", "--merge"])
    assert "projected 0.00h for 30 steps" in capsys.readouterr().out  # resumed at 60, not restarted
    assert json.loads((out / "checkpoint-90" / "trainer_state.json").read_text())["global_step"] == 90
    assert sum(v["n"] for g in report.values() for v in g.values()) > 0
    merged = AutoModelForCausalLM.from_pretrained(out / "merged")
    assert merged.config.vocab_size == 32016
    trained = load_file(out / "adapter" / "adapter_model.safetensors")
    lm_head = next(v for k, v in trained.items() if "lm_head" in k)
    assert torch.equal(merged.lm_head.weight.float(), lm_head.to(torch.bfloat16).float())
