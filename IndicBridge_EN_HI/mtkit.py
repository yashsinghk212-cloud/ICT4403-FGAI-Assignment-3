"""mtkit.py - shared toolkit for the machine-translation comparison projects.

Stages: data preparation -> four models (Transformer / mT5 / LLaMA / Gemma) ->
evaluation (BLEU, chrF, time, params) -> temperature study -> responsible-AI audit -> report.
Heavy libraries (torch, transformers) are imported lazily inside the functions that need them.
"""
import gc
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

MODEL_ORDER = ["transformer", "mt5", "llama", "gemma"]
MODEL_ARCH = {"transformer": "Encoder-decoder", "mt5": "Encoder-decoder",
              "llama": "Decoder-only", "gemma": "Decoder-only"}


# ----------------------------------------------------------------------------- helpers
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def norm_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def nwords(s):
    return len(s.split())


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    return time.perf_counter()


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def environment_info():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    except ImportError:
        pass
    for lib in ("transformers", "datasets", "peft", "sacrebleu", "sentencepiece"):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:
            pass
    return info


# ----------------------------------------------------------------------------- data
def prepare_data(cfg, root, force=False):
    """Download, clean, de-duplicate, split (train/val/test) and describe the parallel corpus."""
    d = ensure_dir(os.path.join(root, "data"))
    res = ensure_dir(os.path.join(root, "results"))
    paths = {k: os.path.join(d, f"{k}.jsonl") for k in ("train", "val", "test")}
    if all(os.path.exists(p) for p in paths.values()) and not force:
        return {k: read_jsonl(p) for k, p in paths.items()}

    from datasets import load_dataset
    args = [cfg["hf_name"]] + ([cfg["hf_config"]] if cfg.get("hf_config") else [])
    ds = load_dataset(*args, split="train")
    ds = ds.shuffle(seed=cfg["seed"]).select(range(min(len(ds), cfg["pool"])))
    s, t = cfg["src_lang"], cfg["tgt_lang"]
    need = cfg["n_train"] + cfg["n_val"] + cfg["n_test"]
    seen, pairs, drops = set(), [], Counter()
    for ex in ds:
        a, b = norm_text(ex["translation"][s]), norm_text(ex["translation"][t])
        wa, wb = nwords(a), nwords(b)
        if not a or not b:
            drops["empty"] += 1
        elif wa < cfg["min_words"] or wa > cfg["max_words"] or wb < cfg["min_words"] or wb > cfg["max_words"] * 1.5:
            drops["length"] += 1
        elif max(wa, wb) / max(min(wa, wb), 1) > 2.5:
            drops["length_ratio"] += 1
        elif a == b:
            drops["copy"] += 1
        elif a.lower() in seen:
            drops["duplicate"] += 1
        else:
            seen.add(a.lower())
            pairs.append({"src": a, "tgt": b})
            if len(pairs) >= need:
                break
    nt, nv = cfg["n_test"], cfg["n_val"]
    data = {"test": pairs[:nt], "val": pairs[nt:nt + nv], "train": pairs[nt + nv:nt + nv + cfg["n_train"]]}
    for k, p in paths.items():
        write_jsonl(p, data[k])

    stats = {"dataset": cfg["hf_name"], "config": cfg.get("hf_config"), "pool_examined": len(ds),
             "dropped": dict(drops), "splits": {}}
    for k, rows in data.items():
        sw = [nwords(r["src"]) for r in rows]
        tw = [nwords(r["tgt"]) for r in rows]
        stats["splits"][k] = {"pairs": len(rows), "avg_src_words": round(float(np.mean(sw)), 2),
                              "avg_tgt_words": round(float(np.mean(tw)), 2),
                              "max_src_words": int(max(sw)), "max_tgt_words": int(max(tw))}
    tr = data["train"]
    stats["train_vocab_src_types"] = len({w for r in tr for w in r["src"].lower().split()})
    stats["train_vocab_tgt_types"] = len({w for r in tr for w in r["tgt"].lower().split()})
    stats["samples"] = tr[:5]
    save_json(os.path.join(res, "dataset_stats.json"), stats)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist([nwords(r["src"]) for r in tr], bins=30, alpha=.6, label=cfg["src_name"])
        ax.hist([nwords(r["tgt"]) for r in tr], bins=30, alpha=.6, label=cfg["tgt_name"])
        ax.set_xlabel("Words per sentence")
        ax.set_ylabel("Sentence pairs")
        ax.set_title("Sentence-length distribution (train)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(res, "length_distribution.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass
    return data


# ----------------------------------------------------------------------------- metrics
def score(hyps, refs, cfg):
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize=cfg["bleu_tokenize"])
    b = bleu.corpus_score(hyps, [refs])
    c = CHRF().corpus_score(hyps, [refs])
    cpp = CHRF(word_order=2).corpus_score(hyps, [refs])
    return {"bleu": round(b.score, 2), "chrf": round(c.score, 2), "chrf_pp": round(cpp.score, 2),
            "bleu_signature": str(bleu.get_signature()), "sacrebleu": sacrebleu.__version__}


def _digits(s):
    m = "".join(str(unicodedata.digit(c, 0)) if c.isdigit() else " " for c in s)
    return Counter(re.findall(r"\d+", m))


def audit(cfg, srcs, hyps, refs):
    """Heuristic responsible-AI audit: possible hallucination / omission / sensitive content."""
    sens = [t.lower() for t in cfg["sensitive_terms"]]
    script = re.compile(cfg["script_regex"]) if cfg.get("script_regex") else None
    rows = []
    for i, (s, h, r) in enumerate(zip(srcs, hyps, refs)):
        flags = []
        ratio = max(nwords(h), 0) / max(nwords(s), 1)
        if not h.strip():
            flags.append("empty_output")
        if ratio > 2.0:
            flags.append("over_generation")
        if 0 < ratio < 0.4:
            flags.append("under_translation")
        if h.strip() == s.strip():
            flags.append("copied_source")
        if _digits(s) - _digits(h):
            flags.append("number_mismatch")
        toks = h.split()
        tri = Counter(zip(toks, toks[1:], toks[2:]))
        if tri and max(tri.values()) >= 3:
            flags.append("repetition_loop")
        if script and h.strip():
            letters = [c for c in h if c.isalpha()]
            if letters and sum(bool(script.match(c)) for c in letters) / len(letters) < 0.5:
                flags.append("wrong_script")
        if cfg.get("entity_check"):
            ents = re.findall(r"(?<=[a-z,;] )[A-Z][a-zA-Z]{2,}", s)
            if any(e not in h for e in ents):
                flags.append("entity_changed_or_dropped")
        low = s.lower()
        hit = [t for t in sens if re.search(r"\b" + re.escape(t), low)]
        rows.append({"idx": i, "source": s, "reference": r, "output": h,
                     "flags": ";".join(flags), "sensitive_terms": ";".join(hit)})
    df = pd.DataFrame(rows)
    n = max(len(df), 1)
    summary = {"n": len(df)}
    for f in ["over_generation", "under_translation", "number_mismatch", "repetition_loop",
              "copied_source", "wrong_script", "entity_changed_or_dropped", "empty_output"]:
        summary[f] = int(df["flags"].str.contains(f).sum())
    summary["any_flag_pct"] = round(float(100 * (df["flags"] != "").sum() / n), 2)
    summary["sensitive_source_sentences"] = int((df["sensitive_terms"] != "").sum())
    return df, summary


# ----------------------------------------------------------------------------- finishing a model run
def finish_model(cfg, root, key, label, translate_fn, data, n_params, n_trainable, train_secs,
                 adaptation, history=None, notes=""):
    import torch
    res = ensure_dir(os.path.join(root, "results"))
    out = ensure_dir(os.path.join(res, "outputs"))
    srcs = [r["src"] for r in data["test"]]
    refs = [r["tgt"] for r in data["test"]]

    t0 = now()
    hyps = translate_fn(srcs, 0.0)
    infer = now() - t0
    sc = score(hyps, refs, cfg)
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # sampling-temperature study (first temp_n test sentences)
    n = min(cfg["temp_n"], len(srcs))
    temp_rows = []
    for T in cfg["temperatures"]:
        h = hyps[:n] if T == 0 else translate_fn(srcs[:n], T)
        s = score(h, refs[:n], cfg)
        temp_rows.append({"model": label, "temperature": T, "bleu": s["bleu"], "chrf": s["chrf"],
                          "distinct_outputs_pct": round(100 * len(set(h)) / max(len(h), 1), 1)})
    # probe sentences for bias / culture / sensitive content
    probes = cfg["probes"]
    probe_out = translate_fn([p[1] for p in probes], 0.0)
    probe_rows = [{"category": c, "source": s, label: o} for (c, s), o in zip(probes, probe_out)]

    adf, asum = audit(cfg, srcs, hyps, refs)
    adf.to_csv(os.path.join(out, f"audit_{key}.csv"), index=False, encoding="utf-8")
    pd.DataFrame({"source": srcs, "reference": refs, "output": hyps}).to_csv(
        os.path.join(out, f"translations_{key}.csv"), index=False, encoding="utf-8")
    save_json(os.path.join(out, f"temperature_{key}.json"), temp_rows)
    save_json(os.path.join(out, f"probes_{key}.json"), probe_rows)

    m = {"key": key, "model": label, "architecture": MODEL_ARCH[key], "adaptation": adaptation,
         "params_total": int(n_params), "params_trainable": int(n_trainable),
         "train_seconds": round(train_secs, 1), "infer_seconds": round(infer, 2),
         "ms_per_sentence": round(1000 * infer / len(srcs), 1), "peak_gpu_gb": round(peak, 2),
         "history": history or [], "audit": asum, "notes": notes, **sc}
    save_json(os.path.join(res, f"metrics_{key}.json"), m)
    print(f"[{label}] BLEU {sc['bleu']} | chrF {sc['chrf']} | train {train_secs/60:.1f} min | "
          f"infer {1000*infer/len(srcs):.0f} ms/sent")
    return m


# ----------------------------------------------------------------------------- mT5
def run_mt5(cfg, data, root):
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    mc = cfg["mt5"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(mc["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(mc["name"])
    prefix = f"translate {cfg['src_name']} to {cfg['tgt_name']}: "

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            x = tok(prefix + r["src"], max_length=mc["max_len"], truncation=True)
            x["labels"] = tok(text_target=r["tgt"], max_length=mc["max_len"], truncation=True)["input_ids"]
            return x

    bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()  # mT5 is unstable in fp16
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(root, "checkpoints", "mt5"), per_device_train_batch_size=mc["batch"],
        per_device_eval_batch_size=mc["batch"] * 2, learning_rate=mc["lr"], num_train_epochs=mc["epochs"],
        warmup_ratio=0.05, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=bf16, logging_steps=50, report_to="none", seed=cfg["seed"])
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(data["train"]),
                             eval_dataset=DS(data["val"]), processing_class=tok,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    t0 = now()
    trainer.train()
    secs = now() - t0
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    model.eval()
    model.to(dev)

    def translate(srcs, temperature=0.0):
        out = [None] * len(srcs)
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
        for i in range(0, len(order), cfg["infer_batch"]):
            idx = order[i:i + cfg["infer_batch"]]
            enc = tok([prefix + srcs[j] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=mc["max_len"]).to(dev)
            kw = dict(max_new_tokens=cfg["max_new_tokens"], num_beams=1, do_sample=temperature > 0)
            if temperature > 0:
                kw.update(temperature=temperature, top_k=0, top_p=1.0)
            with torch.no_grad():
                g = model.generate(**enc, **kw)
            for j, o in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
                out[j] = o.strip()
        return out

    n_params = sum(p.numel() for p in model.parameters())
    m = finish_model(cfg, root, "mt5", "mT5-small", translate, data, n_params, n_params, secs,
                     "Full fine-tuning", history, f"{mc['name']}, bf16={bf16}")
    del trainer, model
    free_gpu()
    return m


# ----------------------------------------------------------------------------- LLaMA / Gemma
def build_prompt(cfg, src, shots=()):
    s, t = cfg["src_name"], cfg["tgt_name"]
    head = f"Translate the following {s} sentences into {t}.\n\n"
    ex = "".join(f"{s}: {a}\n{t}: {b}\n\n" for a, b in shots)
    return f"{head}{ex}{s}: {src}\n{t}:"


def run_decoder(cfg, data, root, key):
    """key in {'llama','gemma'}. mode 'lora' = LoRA fine-tuning, 'fewshot' = in-context prompting."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    dc = cfg[key]
    label = dc["label"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    tok = AutoTokenizer.from_pretrained(dc["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(dc["name"], torch_dtype=dtype).to(dev)
    secs, history, shots = 0.0, [], []

    if dc["mode"] == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=dc["lora_r"], lora_alpha=2 * dc["lora_r"], lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():          # keep the adapter weights in fp32 for stable updates
            if p.requires_grad:
                p.data = p.data.float()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())

        def encode(r):
            p = tok(build_prompt(cfg, r["src"]), add_special_tokens=True)["input_ids"]
            t = tok(" " + r["tgt"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p + t)[: dc["max_seq"]]
            lab = ([-100] * len(p) + t)[: dc["max_seq"]]
            return {"input_ids": ids, "labels": lab}

        class DS(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.items = [encode(r) for r in rows]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                return self.items[i]

        def collate(batch):
            m = max(len(b["input_ids"]) for b in batch)
            ids = torch.tensor([b["input_ids"] + [tok.pad_token_id] * (m - len(b["input_ids"])) for b in batch])
            lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
            return {"input_ids": ids, "labels": lab, "attention_mask": (torch.arange(m)[None, :] < torch.tensor(
                [len(b["input_ids"]) for b in batch])[:, None]).long()}

        train_rows = data["train"][: dc["train_n"]]
        args = TrainingArguments(
            output_dir=os.path.join(root, "checkpoints", key), per_device_train_batch_size=dc["batch"],
            per_device_eval_batch_size=dc["batch"], gradient_accumulation_steps=dc["grad_accum"],
            learning_rate=dc["lr"], num_train_epochs=dc["epochs"], warmup_ratio=0.03,
            lr_scheduler_type="cosine", eval_strategy="epoch", save_strategy="no",
            bf16=dtype == torch.bfloat16, logging_steps=25, report_to="none", seed=cfg["seed"],
            remove_unused_columns=False)
        trainer = Trainer(model=model, args=args, train_dataset=DS(train_rows),
                          eval_dataset=DS(data["val"][:300]), data_collator=collate)
        t0 = now()
        trainer.train()
        secs = now() - t0
        history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
        model.save_pretrained(os.path.join(root, "checkpoints", key + "_adapter"))
        model = model.merge_and_unload()
        adaptation = f"LoRA fine-tuning (r={dc['lora_r']}, {len(train_rows)} pairs, {dc['epochs']} epoch)"
    else:
        n_trainable, n_total = 0, sum(p.numel() for p in model.parameters())
        rng = random.Random(cfg["seed"])
        cand = [r for r in data["train"] if 6 <= nwords(r["src"]) <= 20]
        shots = [(r["src"], r["tgt"]) for r in rng.sample(cand, dc["k_shot"])]
        adaptation = f"{dc['k_shot']}-shot prompting (no weight updates)"
    model.eval()
    tok.padding_side = "left"

    def translate(srcs, temperature=0.0):
        out = [None] * len(srcs)
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
        bs = cfg["infer_batch"]
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([build_prompt(cfg, srcs[j], shots) for j in idx], return_tensors="pt",
                      padding=True).to(dev)
            kw = dict(max_new_tokens=cfg["max_new_tokens"], pad_token_id=tok.pad_token_id,
                      eos_token_id=tok.eos_token_id, do_sample=temperature > 0,
                      stop_strings=["\n"], tokenizer=tok)
            if temperature > 0:
                kw.update(temperature=temperature, top_k=0, top_p=1.0)
            else:
                kw.update(temperature=None, top_k=None, top_p=None)
            with torch.no_grad():
                g = model.generate(**enc, **kw)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, o in zip(idx, texts):
                out[j] = o.strip().split("\n")[0].strip()
        return out

    m = finish_model(cfg, root, key, label, translate, data, n_total, n_trainable, secs, adaptation,
                     history, f"{dc['name']}, dtype={str(dtype).split('.')[-1]}")
    del model
    free_gpu()
    return m


# ----------------------------------------------------------------------------- report
def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def build_report(cfg, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = os.path.join(root, "results")
    ms = [load_json(os.path.join(res, f"metrics_{k}.json")) for k in MODEL_ORDER
          if os.path.exists(os.path.join(res, f"metrics_{k}.json"))]
    if not ms:
        raise SystemExit("No model results found - run the training stage first.")
    df = pd.DataFrame([{
        "Model": m["model"], "Architecture": m["architecture"], "Adaptation": m["adaptation"],
        "Params (M)": round(m["params_total"] / 1e6, 1), "Trainable (M)": round(m["params_trainable"] / 1e6, 2),
        "BLEU": m["bleu"], "chrF": m["chrf"], "chrF++": m["chrf_pp"],
        "Train/FT time (min)": round(m["train_seconds"] / 60, 1), "Inference (s)": m["infer_seconds"],
        "ms/sentence": m["ms_per_sentence"], "Peak GPU (GB)": m["peak_gpu_gb"]} for m in ms])
    df.to_csv(os.path.join(res, "metrics_summary.csv"), index=False)
    labels = df["Model"].tolist()

    def bars(cols, title, fn, ylabel):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(np.arange(len(labels)) + i * w, df[c], w, label=c)
        ax.set_xticks(np.arange(len(labels)) + w * (len(cols) - 1) / 2)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(res, fn), dpi=150)
        plt.close(fig)

    bars(["BLEU", "chrF"], "Translation quality on the test set", "quality.png", "Score")
    bars(["Train/FT time (min)", "ms/sentence"], "Compute cost", "compute.png", "Minutes / ms")
    bars(["Params (M)", "Trainable (M)"], "Model size", "params.png", "Millions")

    # loss curves
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in ms:
        h = m["history"]
        if not h:
            continue
        if "epoch" in h[0] and "train_loss" in h[0]:
            ax.plot([x["epoch"] for x in h], [x["val_loss"] for x in h], marker="o", label=m["model"] + " (val)")
        else:
            ev = [x for x in h if "eval_loss" in x]
            ax.plot([x["epoch"] for x in ev], [x["eval_loss"] for x in ev], marker="o", label=m["model"] + " (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss (loss definitions differ per model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(res, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # temperature
    trows = []
    for m in ms:
        p = os.path.join(res, "outputs", f"temperature_{m['key']}.json")
        if os.path.exists(p):
            trows += load_json(p)
    tdf = pd.DataFrame(trows)
    if len(tdf):
        tdf.to_csv(os.path.join(res, "temperature_analysis.csv"), index=False)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for mod, g in tdf.groupby("model", sort=False):
            axs[0].plot(g["temperature"], g["bleu"], marker="o", label=mod)
            axs[1].plot(g["temperature"], g["chrf"], marker="o", label=mod)
        axs[0].set_title("BLEU vs sampling temperature")
        axs[1].set_title("chrF vs sampling temperature")
        for a in axs:
            a.set_xlabel("Temperature (0 = greedy)")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(res, "temperature_analysis.png"), dpi=150)
        plt.close(fig)

    # samples + probes
    tr = {m["key"]: pd.read_csv(os.path.join(res, "outputs", f"translations_{m['key']}.csv")) for m in ms}
    first = tr[ms[0]["key"]]
    samp = pd.DataFrame({"Source": first["source"][:6], "Reference": first["reference"][:6]})
    for m in ms:
        samp[m["model"]] = tr[m["key"]]["output"][:6].values
    pdf = None
    for m in ms:
        pj = pd.DataFrame(load_json(os.path.join(res, "outputs", f"probes_{m['key']}.json")))
        pdf = pj if pdf is None else pdf.merge(pj[["source", m["model"]]], on="source")
    pdf.to_csv(os.path.join(res, "probe_translations.csv"), index=False, encoding="utf-8")

    aud = pd.DataFrame([{"Model": m["model"], "Any flag (%)": m["audit"]["any_flag_pct"],
                         "Over-gen.": m["audit"]["over_generation"],
                         "Under-transl.": m["audit"]["under_translation"],
                         "Number mismatch": m["audit"]["number_mismatch"],
                         "Repetition": m["audit"]["repetition_loop"],
                         "Copied source": m["audit"]["copied_source"],
                         "Wrong script": m["audit"]["wrong_script"],
                         "Entity changed": m["audit"]["entity_changed_or_dropped"]} for m in ms])

    best_b = df.loc[df["BLEU"].idxmax()]
    best_c = df.loc[df["chrF"].idxmax()]
    fast = df.loc[df["ms/sentence"].idxmin()]
    ed = df[df["Architecture"] == "Encoder-decoder"]
    do = df[df["Architecture"] == "Decoder-only"]
    lines = [f"**Best BLEU:** {best_b['Model']} ({best_b['BLEU']}). **Best chrF:** {best_c['Model']} ({best_c['chrF']}). "
             f"**Fastest inference:** {fast['Model']} ({fast['ms/sentence']} ms/sentence)."]
    if len(ed) and len(do):
        lines.append(f"Averaged over the two encoder-decoder models, BLEU is {ed['BLEU'].mean():.2f} and chrF "
                     f"{ed['chrF'].mean():.2f}; averaged over the two decoder-only models, BLEU is "
                     f"{do['BLEU'].mean():.2f} and chrF {do['chrF'].mean():.2f}. Mean inference time is "
                     f"{ed['ms/sentence'].mean():.0f} ms/sentence (encoder-decoder) versus "
                     f"{do['ms/sentence'].mean():.0f} ms/sentence (decoder-only).")
    overall = df.assign(rank=df["BLEU"].rank(ascending=False) + df["chrF"].rank(ascending=False)).sort_values("rank")
    lines.append(f"**Overall best-performing model (average rank of BLEU and chrF): {overall.iloc[0]['Model']}.**")

    body = [
        "### Results on the held-out test set", "",
        _md_table(df), "",
        "![quality](results/quality.png)", "",
        "![compute](results/compute.png)", "",
        "![params](results/params.png)", "",
        "![loss](results/loss_curves.png)", "",
        "### Automatic summary", "", "\n\n".join(lines), "",
        "### Sampling-temperature analysis", "",
        f"BLEU/chrF on the first {cfg['temp_n']} test sentences at temperatures {cfg['temperatures']} "
        "(0 = greedy decoding; one sampled run per temperature, seed fixed).", "",
        _md_table(tdf) if len(tdf) else "_not available_", "",
        "![temperature](results/temperature_analysis.png)", "",
        "### Sample translations (first test sentences)", "", _md_table(samp), "",
        "### Responsible-AI audit (heuristic flags on the test set)", "",
        _md_table(aud), "",
        "Flags are automatic heuristics (length ratio, number/entity preservation, repetition, script) that "
        "*point to* possible hallucination or omission; they must be confirmed by reading the rows in "
        "`results/outputs/audit_<model>.csv`.", "",
        "### Probe sentences (bias, culture, sensitive content)", "", _md_table(pdf), "",
    ]
    text = "\n".join(body)
    with open(os.path.join(res, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    cheap = df.loc[df["Train/FT time (min)"].idxmin()]
    concl = (f"The best-performing model on this test set is **{overall.iloc[0]['Model']}** "
             f"(BLEU {overall.iloc[0]['BLEU']}, chrF {overall.iloc[0]['chrF']}), ranked first on the average of BLEU and "
             f"chrF rank. The highest BLEU belongs to {best_b['Model']} and the highest chrF to {best_c['Model']}. "
             f"{fast['Model']} is the fastest at inference ({fast['ms/sentence']} ms/sentence) and "
             f"{cheap['Model']} needed the least training/adaptation time ({cheap['Train/FT time (min)']} min). "
             f"Compare these figures with the model sizes in the results table when weighing quality against cost.")
    rp = os.path.join(root, "README.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            rd = f.read()
        for a, b, txt in (("<!-- RESULTS_START -->", "<!-- RESULTS_END -->", text),
                          ("<!-- CONCLUSION_START -->", "<!-- CONCLUSION_END -->", concl)):
            if a in rd and b in rd:
                rd = rd[:rd.index(a) + len(a)] + "\n" + txt + "\n" + rd[rd.index(b):]
        with open(rp, "w", encoding="utf-8") as f:
            f.write(rd)
    print("Report written to results/report.md and injected into README.md")


# ----------------------------------------------------------------------------- command line
def cli(cfg, root):
    import argparse
    import copy
    ap = argparse.ArgumentParser(description=cfg["project_name"])
    ap.add_argument("--stage", choices=["data", "train", "report", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run (minutes, poor scores)")
    ap.add_argument("--force", action="store_true", help="re-run models that already have results")
    a = ap.parse_args()
    cfg = copy.deepcopy(cfg)
    if a.quick:
        cfg.update(n_train=2000, n_val=100, n_test=100, temp_n=30, pool=20000)
        cfg["transformer"].update(epochs=2, vocab=1000)
        cfg["mt5"].update(epochs=1)
        for k in ("llama", "gemma"):
            cfg[k].update(train_n=500, epochs=1)
        root = ensure_dir(os.path.join(root, "quick_run"))
    ensure_dir(os.path.join(root, "results"))
    save_json(os.path.join(root, "results", "environment.json"), environment_info())
    set_seed(cfg["seed"])
    data = prepare_data(cfg, root, force=a.force and a.stage in ("data", "all"))
    print({k: len(v) for k, v in data.items()})
    if a.stage in ("train", "all"):
        for key in a.models:
            if os.path.exists(os.path.join(root, "results", f"metrics_{key}.json")) and not a.force:
                print(f"[{key}] results exist - skipping (use --force to redo)")
                continue
            if key == "transformer":
                from mt_transformer import run_transformer
                run_transformer(cfg, data, root, sys.modules[__name__])
            elif key == "mt5":
                run_mt5(cfg, data, root)
            else:
                run_decoder(cfg, data, root, key)
    if a.stage in ("report", "all"):
        build_report(cfg, root)
