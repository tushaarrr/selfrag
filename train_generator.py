"""Phase 3: LoRA-train a Self-RAG generator on a seeded subset of the released train.jsonl (PLAN-revised Phase 3).

Same as reference/self-rag/retrieval_lm/finetune.py @ 1fcdc42 in prompt format, token ids and label masking, including
the paragraph-mask bug (#13). Differs where the plan says: bf16 LoRA on one GPU; new-token rows mean-initialised and
trainable in both embed_tokens and lm_head (#10); micro-batch > 1 with length grouping; no processed.json dump.

  train:   python train_generator.py --out runs/gen30k --lr 1e-4 --merge
  pilot:   python train_generator.py --out runs/pilot_lr1e-4 --lr 1e-4 --max_steps 100
  released model on the same held-out rows:
           python train_generator.py --out runs/released_reflect --eval_model selfrag/selfrag_llama2_7b
"""
import argparse
import contextlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                          TrainingArguments)

from eval_selfrag import GRD, REL, ROOT, TOK, UT, sha256, write_manifest

TRAIN = ROOT / "data" / "hf" / "selfrag_train_data" / "train.jsonl"
TOKENIZER = ROOT / "data" / "hf" / "selfrag_llama2_7b"  # Llama-2 sentencepiece + the 16 added tokens, ids 32000-32015
PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
N_BASE = 32000
GROUPS = {"retrieve": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"], "isrel": REL, "issup": GRD,
          "isuse": UT}


def split(n_rows, n_train, heldout_frac, seed):
    """Held-out rows first, then the training subset, from one seeded permutation: disjoint by construction."""
    perm = list(range(n_rows))
    random.Random(seed).shuffle(perm)
    n_held = round(n_rows * heldout_frac)
    return perm[n_held:n_held + n_train], perm[:n_held]


def encode(ex, tok, max_len):
    """encode_with_prompt_completion_format (finetune.py:250-292) with context_markups on."""
    src = (PROMPT_INPUT if ex.get("input", "") != "" else PROMPT_NO_INPUT).format_map(ex)
    ids = tok(src + ex["output"] + tok.eos_token, max_length=max_len, truncation=True).input_ids
    src_len = len(tok(src, max_length=max_len, truncation=True).input_ids)
    labels = list(ids)
    labels[:src_len - 1] = [-100] * (src_len - 1)
    # finetune.py:271-284 has no break: each mask runs from a <paragraph> to the LAST </paragraph> (#13)
    for j in range(src_len, len(labels)):
        if labels[j] == TOK["<paragraph>"]:
            ends = [k for k in range(j, len(labels)) if labels[k] == TOK["</paragraph>"]]
            end = ends[-1] if ends else len(labels) - 1
            labels[j + 1:end] = [-100] * max(0, end - j - 1)
    return dict(input_ids=ids, labels=labels, attention_mask=[1] * len(ids))


def new_rows(model):
    rows = {n: p for n, p in model.named_parameters() if "modules_to_save" in n and p.requires_grad}
    assert len(rows) == 2, f"expected trainable embed_tokens and lm_head copies, got {list(rows)}"
    return rows


@torch.no_grad()
def reflection_accuracy(model, rows, bs=4):
    """Acceptance checks 2-3: teacher-forced top-1 at every reflection-token position in the response, per class.
    Counts positions inside masked paragraph spans too; the released model was trained with the same mask."""
    name = {i: t for t, i in TOK.items()}
    group_of = {TOK[t]: g for g, ts in GROUPS.items() for t in ts}
    conf = defaultdict(Counter)  # gold token -> predicted token ("other" if outside the gold's group)
    model.eval()
    dev = next(model.parameters()).device
    for s in range(0, len(rows), bs):
        batch = rows[s:s + bs]
        width = max(len(r["input_ids"]) for r in batch)
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        att = torch.zeros_like(ids)
        for b, r in enumerate(batch):
            ids[b, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
            att[b, :len(r["input_ids"])] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext():
            pred = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, :-1].argmax(-1).cpu()
        for b, r in enumerate(batch):
            start = next(i for i, x in enumerate(r["labels"]) if x != -100)  # = source length - 1
            for t in range(start, len(r["input_ids"]) - 1):
                gold = r["input_ids"][t + 1]
                if gold in group_of:
                    p = pred[b, t].item()
                    conf[name[gold]][name[p] if group_of.get(p) == group_of[gold] else "other"] += 1
    return {g: {t: dict(n=sum(conf[t].values()), acc=round(conf[t][t] / max(1, sum(conf[t].values())), 4),
                        pred=dict(conf[t])) for t in ts} for g, ts in GROUPS.items()}


class Checks(TrainerCallback):
    """Acceptance check 1 at step 50; stop if the first 200 steps project more than 1.5x the budget."""

    def __init__(self, model, budget_h):
        self.model, self.budget_h = model, budget_h
        self.init = {n: p[N_BASE:].detach().float().cpu().clone() for n, p in new_rows(model).items()}

    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.step0 = time.time(), state.global_step

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 50:
            for n, p in new_rows(self.model).items():
                delta = p[N_BASE:].detach().float().cpu() - self.init[n]
                moved = int((delta.norm(dim=1) > 0).sum())
                print(f"step 50: {n} new rows |delta|={delta.norm():.4g}, {moved}/{len(delta)} rows moved", flush=True)
                assert delta.norm() > 0, f"new-token rows of {n} did not move"
        if state.global_step == min(self.step0 + 200, state.max_steps):
            done = state.global_step - self.step0
            projected_h = (time.time() - self.t0) / done * (state.max_steps - self.step0) / 3600
            print(f"projected {projected_h:.2f}h for {state.max_steps - self.step0} steps (budget {self.budget_h}h)",
                  flush=True)
            if projected_h > 1.5 * self.budget_h:
                print("over 1.5x budget: stopping, ask before continuing", flush=True)
                control.should_training_stop = True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--eval_model", help="no training: reflection accuracy of this model on the held-out rows")
    ap.add_argument("--train_file", default=str(TRAIN))
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--heldout_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4, help="pick from the 1e-4 vs 2e-4 pilot")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1, help="100 for the lr pilot")
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16, help="micro_bs * accum = 128, the paper's batch")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--budget_h", type=float, default=6.4, help="the plan's upper estimate for 30k x 3 epochs")
    ap.add_argument("--merge", action="store_true", help="also save a merged bf16 model; vLLM 0.2.6 has no LoRA")
    a = ap.parse_args(argv)
    out = Path(a.out)
    write_manifest(out / "manifest.json", args=vars(a), train_sha256=sha256(a.train_file))

    lines = open(a.train_file).readlines()
    train_idx, held_idx = split(len(lines), a.n_train, a.heldout_frac, a.seed)
    (out / "split.json").write_text(json.dumps(dict(seed=a.seed, train=train_idx, heldout=held_idx)))
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False)  # the reference trains with --use_slow_tokenizer
    assert len(tok) == 32016 and all(tok.convert_tokens_to_ids(t) == i for t, i in TOK.items())

    def enc(idx):  # the reference drops rows whose labels are all masked (finetune.py:515)
        return [e for e in (encode(json.loads(lines[i]), tok, a.max_len) for i in idx) if set(e["labels"]) != {-100}]

    held = enc(held_idx)
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    if a.eval_model:
        model = AutoModelForCausalLM.from_pretrained(a.eval_model, torch_dtype=dtype).to("cuda" if cuda else "cpu")
        report = reflection_accuracy(model, held)
        (out / "reflection_acc.json").write_text(json.dumps(report, indent=1))
        return report

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype)
    model.config.use_cache = False
    model.resize_token_embeddings(len(tok))
    for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):  # untied in Llama-2
        w.data[N_BASE:] = w.data[:N_BASE].mean(0)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"]))  # the reference leaves this commented out (#10)
    for p in model.parameters():  # fp32 master weights for everything trainable; bf16 compute via autocast
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    collate = DataCollatorForSeq2Seq(tok, padding="longest")
    trainer = Trainer(
        model=model, train_dataset=enc(train_idx),
        data_collator=lambda feats: collate([dict(f) for f in feats]),  # it pads labels in place; epoch 2 would crash
        callbacks=[Checks(model, a.budget_h)],
        args=TrainingArguments(
            output_dir=str(out), per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=a.accum,
            learning_rate=a.lr, lr_scheduler_type="linear", warmup_ratio=0.03, weight_decay=0.0,
            num_train_epochs=a.epochs, max_steps=a.max_steps, bf16=cuda, group_by_length=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10, save_steps=a.save_steps, save_total_limit=2, report_to="none", seed=a.seed,
            remove_unused_columns=False))
    trainer.train(resume_from_checkpoint=True if any(out.glob("checkpoint-*")) else None)
    trainer.save_model(str(out / "adapter"))
    report = reflection_accuracy(model, held)
    (out / "reflection_acc.json").write_text(json.dumps(report, indent=1))
    if a.merge:
        merged = model.merge_and_unload().cpu().to(torch.bfloat16)
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
    return report


if __name__ == "__main__":
    main()
