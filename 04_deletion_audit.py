from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

@dataclass
class A:
    tokens: List[str]
    labels: List[str]
    sent_id: str = ""

@dataclass
class B:
    split: str
    sent_idx: int
    word_idx: int
    token: str
    label: str
    label_id: int
    hidden: np.ndarray
    loss: float
    pred_id: int
    correct: bool
    prob: float
    def uid(self) -> str:
        return f"{self.split}:{self.sent_idx}:{self.word_idx}"

class C:
    def __init__(self, xs: List[A], tok: Any, m: Dict[str, int], max_length: int = 256, drop: Optional[set] = None) -> None:
        self.xs = xs
        self.tok = tok
        self.m = m
        self.max_length = max_length
        self.drop = drop or set()
        self.enc = []
        self.pos = []
        self._run()
    def _run(self) -> None:
        for si, x in enumerate(self.xs):
            z = self.tok(x.tokens, is_split_into_words=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            w = z.word_ids(batch_index=0)
            y = []
            p = {}
            seen = set()
            for j, wi in enumerate(w):
                if wi is None:
                    y.append(-100)
                elif wi not in seen:
                    seen.add(wi)
                    p[wi] = j
                    if wi >= len(x.labels) or (si, wi) in self.drop:
                        y.append(-100)
                    else:
                        y.append(self.m[x.labels[wi]])
                else:
                    y.append(-100)
            z["labels"] = torch.tensor([y], dtype=torch.long)
            self.enc.append(z)
            self.pos.append(p)
    def __len__(self) -> int:
        return len(self.xs)

def rj(path: str) -> List[A]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            o = json.loads(s)
            t = o.get("tokens")
            y = o.get("labels") or o.get("srl_labels")
            if t is None or y is None or len(t) != len(y):
                raise ValueError(str(i + 1))
            out.append(A(list(t), list(y), str(o.get("sent_id", i))))
    return out

def rc(path: str, token_col: int = 0, label_col: int = -1) -> List[A]:
    out = []
    t = []
    y = []
    k = 0
    def flush():
        nonlocal t, y, k
        if t:
            out.append(A(t, y, str(k)))
            k += 1
            t = []
            y = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                flush()
                continue
            if s.startswith("#"):
                continue
            p = s.split()
            t.append(p[token_col])
            y.append(p[label_col])
    flush()
    return out

def load(path: str, fmt: str, tc: int, lc: int) -> List[A]:
    return rj(path) if fmt == "jsonl" else rc(path, tc, lc)

def maps(*cc: List[A]) -> Tuple[Dict[str, int], Dict[int, str]]:
    z = sorted({a for c in cc for x in c for a in x.labels})
    m = {a: i for i, a in enumerate(z)}
    im = {i: a for a, i in m.items()}
    return m, im

def mb(b: Dict[str, Tensor], d: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(d) for k, v in b.items()}

def gp(model: torch.nn.Module, mode: str) -> List[torch.nn.Parameter]:
    ps = []
    ns = list(model.named_parameters())
    if mode == "classifier":
        for n, p in ns:
            if p.requires_grad and ("classifier" in n or "score" in n or "crf" in n):
                ps.append(p)
    elif mode == "encoder_last":
        for n, p in ns:
            if p.requires_grad and ("classifier" in n or "score" in n or "encoder.layer.11" in n or "bert.encoder.layer.11" in n):
                ps.append(p)
    elif mode == "all":
        ps = [p for _, p in ns if p.requires_grad]
    if not ps:
        raise ValueError("no parameters")
    return ps

def freeze_for_head(model: torch.nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = ("classifier" in n or "score" in n or "crf" in n)

def collate(xs: List[Dict[str, Tensor]], pad: int) -> Dict[str, Tensor]:
    mx = max(x["input_ids"].shape[1] for x in xs)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in xs:
        n = x["input_ids"].shape[1]
        q = mx - n
        out["input_ids"].append(F.pad(x["input_ids"][0], (0, q), value=pad))
        out["attention_mask"].append(F.pad(x["attention_mask"][0], (0, q), value=0))
        out["labels"].append(F.pad(x["labels"][0], (0, q), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}

@torch.no_grad()
def recs(split: str, c: C, model: torch.nn.Module, dev: torch.device, im: Dict[int, str], layer: int) -> List[B]:
    model.eval()
    out = []
    for si in tqdm(range(len(c)), desc=f"x-{split}"):
        b = mb(c.enc[si], dev)
        o = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), token_type_ids=b.get("token_type_ids", None), labels=None, output_hidden_states=True, return_dict=True)
        logits = o.logits[0]
        hs = o.hidden_states[layer][0]
        labs = b["labels"][0]
        probs = torch.softmax(logits, dim=-1)
        for wi, pj in c.pos[si].items():
            if wi >= len(c.xs[si].tokens):
                continue
            yi = int(labs[pj].item())
            if yi == -100:
                continue
            li = F.cross_entropy(logits[pj].view(1, -1), torch.tensor([yi], device=dev))
            pi = int(torch.argmax(logits[pj]).item())
            out.append(B(split, si, wi, c.xs[si].tokens[wi], im[yi], yi, hs[pj].detach().cpu().float().numpy(), float(li.detach().cpu()), pi, pi == yi, float(probs[pj, yi].detach().cpu())))
    return out

def norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)

def lcts(tr: List[B], te: List[B], eps: float = 1e-6):
    h1 = np.stack([x.hidden for x in tr]).astype(np.float32)
    h2 = np.stack([x.hidden for x in te]).astype(np.float32)
    dim = h1.shape[1]
    by = defaultdict(list)
    for i, x in enumerate(tr):
        by[x.label].append(i)
    st = {}
    for y, idx in by.items():
        h = h1[idx]
        c = h.mean(0)
        r = h - c
        mu = r.mean(0)
        sd = r.std(0)
        sd = np.where(sd < eps, 1.0, sd)
        st[y] = (c, mu, sd)
    gc = h1.mean(0)
    gm = np.zeros(dim, dtype=np.float32)
    gs = np.ones(dim, dtype=np.float32)
    def tx(rr, hh):
        z = np.zeros_like(hh, dtype=np.float32)
        for i, x in enumerate(rr):
            c, mu, sd = st.get(x.label, (gc, gm, gs))
            z[i] = (hh[i] - c - mu) / sd
        return z
    return tx(tr, h1), tx(te, h2)

def ret_lcts(tr: List[B], te: List[B], z1: np.ndarray, z2: np.ndarray, k: int):
    by = defaultdict(list)
    for i, x in enumerate(tr):
        by[x.label].append(i)
    out = {}
    for ti, x in enumerate(tqdm(te, desc="r0")):
        idx = by.get(x.label, [])
        if not idx:
            out[ti] = []
            continue
        d = np.linalg.norm(z1[idx] - z2[ti], axis=1)
        q = np.argsort(d)[:k]
        out[ti] = [idx[int(j)] for j in q]
    return out

def ret_raw(tr: List[B], te: List[B], k: int):
    h1 = np.stack([x.hidden for x in tr]).astype(np.float32)
    h2 = np.stack([x.hidden for x in te]).astype(np.float32)
    by = defaultdict(list)
    for i, x in enumerate(tr):
        by[x.label].append(i)
    out = {}
    for ti, x in enumerate(tqdm(te, desc="r1")):
        idx = by.get(x.label, [])
        if not idx:
            out[ti] = []
            continue
        d = 1.0 - np.matmul(norm(h2[ti:ti+1]), norm(h1[idx]).T)[0]
        q = np.argsort(d)[:k]
        out[ti] = [idx[int(j)] for j in q]
    return out

class G:
    def __init__(self, model, c1, c2, dev, ps):
        self.model = model
        self.c1 = c1
        self.c2 = c2
        self.dev = dev
        self.ps = ps
        self.cache = {}
    def c(self, s):
        return self.c1 if s == "train" else self.c2
    def g(self, x: B) -> Tensor:
        u = x.uid()
        if u in self.cache:
            return self.cache[u]
        c = self.c(x.split)
        pos = c.pos[x.sent_idx][x.word_idx]
        b = mb(c.enc[x.sent_idx], self.dev)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        o = self.model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), token_type_ids=b.get("token_type_ids", None), labels=None, return_dict=True)
        logits = o.logits[0, pos]
        loss = F.cross_entropy(logits.view(1, -1), torch.tensor([x.label_id], device=self.dev))
        gs = torch.autograd.grad(loss, self.ps, retain_graph=False, create_graph=False, allow_unused=True)
        flat = []
        for a, p in zip(gs, self.ps):
            flat.append(torch.zeros(p.numel(), dtype=torch.float32) if a is None else a.detach().cpu().float().flatten())
        z = torch.cat(flat)
        self.cache[u] = z
        return z

def cos(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    d = torch.norm(a) * torch.norm(b)
    if float(d) < eps:
        return 0.0
    return float(torch.dot(a, b) / d)

def sc(a: B, b: B, ga: Tensor, gb: Tensor, mode: str) -> float:
    if mode == "if":
        return math.exp(-a.loss) * cos(ga, gb)
    return float(torch.dot(ga, gb))

def pick(tr: List[B], te: List[B], chosen: List[int], ret: Dict[int, List[int]], gr: G, mode: str, seed: int):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, x in enumerate(tr):
        by[x.label].append(i)
    out = {}
    for ti in tqdm(chosen, desc=f"p-{mode}"):
        x = te[ti]
        if mode == "rand":
            pool = by.get(x.label, [])
            if not pool:
                continue
            out[ti] = rng.choice(pool)
            continue
        cand = ret.get(ti, [])
        if not cand:
            continue
        gx = gr.g(x)
        best = None
        bv = -1e30
        for ci in cand:
            y = tr[ci]
            v = sc(y, x, gr.g(y), gx, "if" if mode == "if" else "dot")
            if v > bv:
                bv = v
                best = ci
        if best is not None:
            out[ti] = best
    return out

def train_head(base: str, tok, m, im, train_x: List[A], drop: set, dev, args):
    model = AutoModelForTokenClassification.from_pretrained(base, num_labels=len(m), id2label=im, label2id=m, ignore_mismatched_sizes=True)
    model.to(dev)
    freeze_for_head(model)
    c = C(train_x, tok, m, args.max_length, drop)
    dl = DataLoader(c.enc, batch_size=args.batch_size, shuffle=True, collate_fn=lambda xs: collate(xs, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.epochs):
        for b in tqdm(dl, desc="fit"):
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"], return_dict=True)
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
    model.eval()
    return model

@torch.no_grad()
def probs_for(model, c: C, dev, selected: List[int], te: List[B]) -> Dict[int, float]:
    model.eval()
    out = {}
    for ti in tqdm(selected, desc="prob"):
        x = te[ti]
        b = mb(c.enc[x.sent_idx], dev)
        p = c.pos[x.sent_idx][x.word_idx]
        o = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"), token_type_ids=b.get("token_type_ids", None), labels=None, return_dict=True)
        pr = torch.softmax(o.logits[0, p], dim=-1)
        out[ti] = float(pr[x.label_id].detach().cpu())
    return out

def pperm(a: np.ndarray, b: np.ndarray, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    d = a - b
    obs = float(np.mean(d))
    if np.allclose(d, 0):
        return 1.0
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], size=d.shape[0])
        if float(np.mean(d * s)) >= obs:
            cnt += 1
    return float((cnt + 1) / (n + 1))

def dz(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    s = float(np.std(d, ddof=1))
    if s == 0:
        return float("nan")
    return float(np.mean(d) / s)

def save(outdir: Path, rows, details):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "deletion_audit_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "mean_cd", "sd", "p_vs_random", "cohens_dz_vs_random", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(outdir / "deletion_audit_table.md", "w", encoding="utf-8") as f:
        f.write("| Deletion target | Mean confidence drop ± SD | p-value vs random | Cohen's dz vs random | n |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            pv = "—" if r["p_vs_random"] == "" else r["p_vs_random"]
            dzv = "—" if r["cohens_dz_vs_random"] == "" else r["cohens_dz_vs_random"]
            f.write(f"| {r['target']} | {float(r['mean_cd']):.4f} ± {float(r['sd']):.4f} | {pv} | {dzv} | {r['n']} |\n")
    with open(outdir / "deletion_audit_details.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--format", choices=["conll", "jsonl"], default="conll")
    ap.add_argument("--token_col", type=int, default=0)
    ap.add_argument("--label_col", type=int, default=-1)
    ap.add_argument("--output_dir", default="outputs/deletion_audit")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--sample_n", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--gradient_params", choices=["classifier", "encoder_last", "all"], default="classifier")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    tr_x = load(args.train_file, args.format, args.token_col, args.label_col)
    te_x = load(args.test_file, args.format, args.token_col, args.label_col)
    m, im = maps(tr_x, te_x)
    tok = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint, num_labels=len(m), id2label=im, label2id=m, ignore_mismatched_sizes=True)
    model.to(dev)
    model.eval()
    tr_c = C(tr_x, tok, m, args.max_length)
    te_c = C(te_x, tok, m, args.max_length)
    tr_r = recs("train", tr_c, model, dev, im, args.layer)
    te_r = recs("test", te_c, model, dev, im, args.layer)
    correct = [i for i, x in enumerate(te_r) if x.correct]
    rng = random.Random(args.seed)
    if len(correct) < args.sample_n:
        raise ValueError(f"only {len(correct)} correctly predicted tokens")
    selected = sorted(rng.sample(correct, args.sample_n))
    z1, z2 = lcts(tr_r, te_r)
    r0 = ret_lcts(tr_r, te_r, z1, z2, args.top_k)
    r1 = ret_raw(tr_r, te_r, args.top_k)
    ps = gp(model, args.gradient_params)
    gr = G(model, tr_c, te_c, dev, ps)
    p_if = pick(tr_r, te_r, selected, r0, gr, "if", args.seed)
    p_tr = pick(tr_r, te_r, selected, r0, gr, "dot", args.seed)
    p_ra = pick(tr_r, te_r, selected, r1, gr, "rand", args.seed)
    keep = [i for i in selected if i in p_if and i in p_tr and i in p_ra]
    if len(keep) < 2:
        raise ValueError("not enough matched samples")
    orig = {i: te_r[i].prob for i in keep}
    strategies = {"Top supportive token by LCTS + IF*": p_if, "Top supportive token by TracIn-style baseline": p_tr, "Random same-label token": p_ra}
    cds = {}
    afters = {}
    for name, mp in strategies.items():
        drop = {(tr_r[mp[i]].sent_idx, tr_r[mp[i]].word_idx) for i in keep}
        nm = train_head(args.model_checkpoint, tok, m, im, tr_x, drop, dev, args)
        nc = C(te_x, tok, m, args.max_length)
        pr = probs_for(nm, nc, dev, keep, te_r)
        afters[name] = pr
        cds[name] = np.array([orig[i] - pr[i] for i in keep], dtype=np.float64)
    rnd = cds["Random same-label token"]
    rows = []
    for name, arr in cds.items():
        if name == "Random same-label token":
            pv = ""
            ez = ""
        else:
            pv = f"{pperm(arr, rnd, args.permutations, args.seed):.6f}"
            ez = f"{dz(arr, rnd):.4f}"
        rows.append({"target": name, "mean_cd": f"{float(np.mean(arr)):.6f}", "sd": f"{float(np.std(arr, ddof=1)):.6f}", "p_vs_random": pv, "cohens_dz_vs_random": ez, "n": len(keep)})
    details = []
    for i in keep:
        row = {"test_uid": te_r[i].uid(), "token": te_r[i].token, "label": te_r[i].label, "original_prob": orig[i]}
        for name in strategies:
            j = strategies[name][i]
            row[name + "_train_uid"] = tr_r[j].uid()
            row[name + "_train_token"] = tr_r[j].token
            row[name + "_after_prob"] = afters[name][i]
            row[name + "_confidence_drop"] = orig[i] - afters[name][i]
        details.append(row)
    save(Path(args.output_dir), rows, details)
    print(Path(args.output_dir) / "deletion_audit_table.md")

if __name__ == "__main__":
    main()
