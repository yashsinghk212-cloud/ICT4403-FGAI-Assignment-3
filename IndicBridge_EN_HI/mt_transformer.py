"""Basic encoder-decoder Transformer (trained from scratch) for machine translation.

Uses a joint SentencePiece BPE vocabulary, a shared embedding matrix (tied with the
output projection), pre-LayerNorm blocks, label smoothing and warm-up + inverse-sqrt LR.
"""
import copy
import math
import os
import time

import torch
import torch.nn as nn
import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, d_model=256, nhead=4, layers=3, ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model ** -0.5)
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.tr = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=ff, dropout=dropout, batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(d_model, vocab, bias=False)
        self.out.weight = self.emb.weight  # weight tying

    def _embed(self, x):
        return self.drop(self.pos(self.emb(x) * math.sqrt(self.d_model)))

    def encode(self, src):
        src_pad = src.eq(PAD)
        memory = self.tr.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt_in, memory, src_pad, tgt_pad=None):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.tr.decoder(
            self._embed(tgt_in), memory, tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad, tgt_in.eq(PAD))

    @torch.no_grad()
    def generate(self, src, max_len=100, temperature=0.0):
        self.eval()
        memory, src_pad = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_pad)[:, -1].float()
            logits[:, PAD] = float("-inf")
            logits[:, BOS] = float("-inf")
            if temperature and temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(EOS)
            if done.all():
                break
        return ys[:, 1:]


def _pad(seqs):
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (m - len(s)) for s in seqs], dtype=torch.long)


def train_sentencepiece(texts, prefix, vocab_size):
    corpus = prefix + "_corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=vocab_size, model_type="bpe",
        character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        hard_vocab_limit=False, input_sentence_size=1_000_000, shuffle_input_sentence=True,
    )
    os.remove(corpus)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def train_transformer(tc, sp, train_pairs, val_pairs, device, seed, log=print):
    """Returns (best_model, history, train_seconds)."""
    torch.manual_seed(seed)
    V = sp.get_piece_size()
    model = Seq2SeqTransformer(V, tc["d_model"], tc["nhead"], tc["layers"], tc["ff"], tc["dropout"]).to(device)
    L = tc["max_len"]

    def enc(pairs):
        s = sp.encode([p[0] for p in pairs])
        t = sp.encode([p[1] for p in pairs])
        return [(a[: L - 1] + [EOS], [BOS] + b[: L - 1], b[: L - 1] + [EOS]) for a, b in zip(s, t)]

    tr, va = enc(train_pairs), enc(val_pairs)

    def batches(data, bs, shuffle):
        idx = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
        for i in range(0, len(idx), bs):
            b = [data[j] for j in idx[i:i + bs]]
            yield _pad([x[0] for x in b]), _pad([x[1] for x in b]), _pad([x[2] for x in b])

    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], betas=(0.9, 0.98), eps=1e-9)
    warm = tc["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, math.sqrt(warm / (s + 1))))
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    use_cuda = device.type == "cuda"
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not bf16)

    history, best, best_val, bad = [], None, float("inf"), 0
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for s, ti, to in batches(tr, tc["batch"], True):
            s, ti, to = s.to(device), ti.to(device), to.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
                loss = crit(model(s, ti).reshape(-1, V).float(), to.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n += 1
        model.eval()
        vt, vn = 0.0, 0
        with torch.no_grad():
            for s, ti, to in batches(va, tc["batch"] * 2, False):
                s, ti, to = s.to(device), ti.to(device), to.to(device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
                    vl = crit(model(s, ti).reshape(-1, V).float(), to.reshape(-1))
                vt += vl.item()
                vn += 1
        h = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vt / max(vn, 1)}
        history.append(h)
        log(f"[transformer] epoch {ep}: train {h['train_loss']:.3f} | val {h['val_loss']:.3f}")
        if h["val_loss"] < best_val - 1e-4:
            best_val, bad = h["val_loss"], 0
            best = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tc["patience"]:
                log("[transformer] early stopping")
                break
    if use_cuda:
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    model.load_state_dict(best)
    return model, history, secs


def make_translate_fn(model, sp, tc, device, batch_size):
    L = tc["max_len"]

    def translate(srcs, temperature=0.0):
        out = [None] * len(srcs)
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            ids = [sp.encode(srcs[j])[: L - 1] + [EOS] for j in idx]
            gen = model.generate(_pad(ids).to(device), max_len=L, temperature=temperature).cpu().tolist()
            for j, g in zip(idx, gen):
                g = g[: g.index(EOS)] if EOS in g else g
                out[j] = sp.decode([t for t in g if t not in (PAD, BOS)])
        return out

    return translate


def run_transformer(cfg, data, root, kit):
    """Full pipeline for the from-scratch Transformer (uses helpers from mtkit)."""
    tc = cfg["transformer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kit.set_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ck = kit.ensure_dir(os.path.join(root, "checkpoints"))
    sp = train_sentencepiece([r["src"] for r in data["train"]] + [r["tgt"] for r in data["train"]],
                             os.path.join(ck, "spm_joint"), tc["vocab"])
    tr = [(r["src"], r["tgt"]) for r in data["train"]]
    va = [(r["src"], r["tgt"]) for r in data["val"]]
    model, hist, secs = train_transformer(tc, sp, tr, va, device, cfg["seed"])
    torch.save(model.state_dict(), os.path.join(ck, "transformer.pt"))
    translate = make_translate_fn(model, sp, tc, device, cfg["infer_batch"])
    n = sum(p.numel() for p in model.parameters())
    desc = (f"{tc['layers']}+{tc['layers']} layers, d_model={tc['d_model']}, heads={tc['nhead']}, "
            f"ff={tc['ff']}, joint BPE vocab={sp.get_piece_size()}")
    m = kit.finish_model(cfg, root, "transformer", "Transformer (scratch)", translate, data, n, n, secs,
                         "Trained from scratch", hist, desc)
    del model
    kit.free_gpu()
    return m
