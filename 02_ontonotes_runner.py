from __future__ import annotations
import argparse
import csv
import math
import random
import re
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
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except Exception as exc:
    raise RuntimeError('Please install transformers: pip install transformers') from exc

@dataclass
class SRLInstance:
    tokens: List[str]
    labels: List[str]
    predicate_index: int
    sent_id: str = ''
    predicate: str = ''

@dataclass
class EncodedInstance:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    original_word_positions: Dict[int, int]

@dataclass
class TokenRecord:
    split: str
    instance_idx: int
    word_idx: int
    token: str
    label: str
    label_id: int
    hidden: np.ndarray
    loss: float
    pred_id: int
    correct: bool

    @property
    def uid(self) -> str:
        return f'{self.split}:{self.instance_idx}:{self.word_idx}'

def normalise_srl_label(label: str) -> str:
    if label is None:
        return 'O'
    label = str(label).strip()
    if not label or label in {'*', '-', '_', 'O'}:
        return 'O'
    if label.startswith('B-') or label.startswith('I-'):
        return label
    return 'B-' + label

def convert_frame_labels_to_iob(labels: List[str]) -> List[str]:
    converted = []
    prev_base = 'O'
    for raw in labels:
        lab = normalise_srl_label(raw)
        if lab == 'O':
            converted.append('O')
            prev_base = 'O'
            continue
        prefix, base = lab.split('-', 1)
        if prefix == 'I' and prev_base != base:
            converted.append('B-' + base)
        else:
            converted.append(lab)
        prev_base = base
    return converted

def parse_conll_srl_column(cells: List[str]) -> List[str]:
    labels: List[str] = []
    active: Optional[str] = None
    for cell in cells:
        cell = cell.strip()
        closed = re.search('\\(([^()*]+)\\*\\)', cell)
        if closed:
            labels.append('B-' + closed.group(1))
            active = None
            continue
        opened = re.search('\\(([^()*]+)\\*', cell)
        if opened:
            role = opened.group(1)
            labels.append('B-' + role)
            active = None if ')' in cell else role
            continue
        if active is not None:
            labels.append('I-' + active)
            if ')' in cell:
                active = None
            continue
        labels.append('O')
    return labels

def read_local_conll2012(path: str) -> List[SRLInstance]:
    instances: List[SRLInstance] = []
    rows: List[List[str]] = []
    sent_counter = 0

    def flush() -> None:
        nonlocal rows, sent_counter
        if not rows:
            return
        tokens = [r[3] for r in rows if len(r) > 11]
        if not tokens:
            rows = []
            return
        min_cols = min((len(r) for r in rows))
        srl_start, srl_end = (11, min_cols - 1)
        pred_lemmas = [r[6] if len(r) > 6 else '-' for r in rows]
        predicate_positions = [i for i, p in enumerate(pred_lemmas) if p not in {'-', '_', ''}]
        if srl_end > srl_start:
            for local_col_idx, col in enumerate(range(srl_start, srl_end)):
                labels = parse_conll_srl_column([r[col] for r in rows])
                if all((x == 'O' for x in labels)):
                    continue
                pred_idx = predicate_positions[local_col_idx] if local_col_idx < len(predicate_positions) else 0
                pred = tokens[pred_idx] if 0 <= pred_idx < len(tokens) else ''
                instances.append(SRLInstance(tokens, labels, pred_idx, f'local:{sent_counter}:pred{local_col_idx}', pred))
        sent_counter += 1
        rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line:
                flush()
                continue
            if line.startswith('#begin') or line.startswith('#end'):
                continue
            parts = line.split()
            if len(parts) > 3:
                rows.append(parts)
        flush()
    return instances

def _extract_hf_instances_from_sentence(sent: Any, sent_id: str) -> List[SRLInstance]:
    instances: List[SRLInstance] = []
    if not isinstance(sent, dict):
        return instances
    tokens = sent.get('words') or sent.get('tokens')
    frames = sent.get('srl_frames') or sent.get('srl') or sent.get('frames')
    if tokens is None or frames is None:
        return instances
    for j, frame in enumerate(frames):
        labels, pred, pred_idx = (None, '', None)
        if isinstance(frame, dict):
            labels = frame.get('frames') or frame.get('tags') or frame.get('labels') or frame.get('srl_tags')
            pred = str(frame.get('verb') or frame.get('predicate') or '')
            pred_idx = frame.get('predicate_index') or frame.get('verb_index')
        elif isinstance(frame, (list, tuple)):
            labels = frame
        if labels is None or len(labels) != len(tokens):
            continue
        labels = convert_frame_labels_to_iob([str(x) for x in labels])
        if pred_idx is None:
            v_pos = [i for i, lab in enumerate(labels) if lab in {'B-V', 'I-V', 'B-PRED', 'I-PRED'}]
            pred_idx = v_pos[0] if v_pos else 0
        pred_idx = int(pred_idx)
        if not pred and 0 <= pred_idx < len(tokens):
            pred = str(tokens[pred_idx])
        if not all((x == 'O' for x in labels)):
            instances.append(SRLInstance([str(t) for t in tokens], labels, pred_idx, f'{sent_id}:frame{j}', pred))
    return instances

def load_hf_ontonotes(dataset_name: str, config_name: Optional[str], split: str) -> List[SRLInstance]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError('Please install datasets: pip install datasets') from exc
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    instances: List[SRLInstance] = []
    for i, row in enumerate(tqdm(ds, desc=f'Reading HF split={split}')):
        if isinstance(row, dict) and 'sentences' in row:
            for s_idx, sent in enumerate(row['sentences']):
                instances.extend(_extract_hf_instances_from_sentence(sent, f'hf:{split}:{i}:sent{s_idx}'))
        else:
            instances.extend(_extract_hf_instances_from_sentence(row, f'hf:{split}:{i}'))
    if not instances:
        raise RuntimeError('No SRL instances were extracted. Try --source conll with local CoNLL-2012 files.')
    return instances

def add_predicate_markers(inst: SRLInstance) -> Tuple[List[str], List[Optional[int]]]:
    aug_tokens: List[str] = []
    aug_to_orig: List[Optional[int]] = []
    for i, tok in enumerate(inst.tokens):
        if i == inst.predicate_index:
            aug_tokens += ['<p>', tok, '</p>']
            aug_to_orig += [None, i, None]
        else:
            aug_tokens.append(tok)
            aug_to_orig.append(i)
    return (aug_tokens, aug_to_orig)

class EncodedSRLCorpus:

    def __init__(self, instances: List[SRLInstance], tokenizer: Any, label2id: Dict[str, int], max_length: int=256) -> None:
        self.instances = instances
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.encoded: List[EncodedInstance] = []
        self.first_subtoken_pos: List[Dict[int, int]] = []
        self._encode()

    def _encode(self) -> None:
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<p>', '</p>']})
        for inst in self.instances:
            aug_tokens, aug_to_orig = add_predicate_markers(inst)
            enc = self.tokenizer(aug_tokens, is_split_into_words=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            word_ids = enc.word_ids(batch_index=0)
            labels: List[int] = []
            first_pos: Dict[int, int] = {}
            seen_aug = set()
            for tok_pos, aug_word_idx in enumerate(word_ids):
                if aug_word_idx is None or aug_word_idx in seen_aug:
                    labels.append(-100)
                    continue
                seen_aug.add(aug_word_idx)
                orig_idx = aug_to_orig[aug_word_idx] if aug_word_idx < len(aug_to_orig) else None
                if orig_idx is None:
                    labels.append(-100)
                else:
                    labels.append(self.label2id[inst.labels[orig_idx]])
                    first_pos[orig_idx] = tok_pos
            self.encoded.append(EncodedInstance(enc['input_ids'][0], enc['attention_mask'][0], torch.tensor(labels, dtype=torch.long), first_pos))
            self.first_subtoken_pos.append(first_pos)

    def __len__(self) -> int:
        return len(self.instances)

def collate_batch(items: List[EncodedInstance], pad_token_id: int) -> Dict[str, Tensor]:
    max_len = max((x.input_ids.shape[0] for x in items))
    ids, masks, labs = ([], [], [])
    for x in items:
        pad = max_len - x.input_ids.shape[0]
        ids.append(F.pad(x.input_ids, (0, pad), value=pad_token_id))
        masks.append(F.pad(x.attention_mask, (0, pad), value=0))
        labs.append(F.pad(x.labels, (0, pad), value=-100))
    return {'input_ids': torch.stack(ids), 'attention_mask': torch.stack(masks), 'labels': torch.stack(labs)}

def build_label_maps(train: List[SRLInstance], test: List[SRLInstance]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({lab for inst in train + test for lab in inst.labels} | {'O'})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return (label2id, id2label)

def train_token_classifier(model: torch.nn.Module, train_corpus: EncodedSRLCorpus, tokenizer: Any, output_dir: Path, device: torch.device, epochs: int, batch_size: int, lr: float) -> torch.nn.Module:
    model.train().to(device)
    loader = DataLoader(train_corpus.encoded, batch_size=batch_size, shuffle=True, collate_fn=lambda xs: collate_batch(xs, tokenizer.pad_token_id))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total, steps = (0.0, 0)
        for batch in tqdm(loader, desc=f'Fine-tuning epoch {epoch + 1}/{epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.detach().cpu())
            steps += 1
        print(f'epoch={epoch + 1}, loss={total / max(steps, 1):.4f}')
    ckpt = output_dir / 'ontonotes_srl_checkpoint'
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    print(f'Saved fine-tuned checkpoint to {ckpt}')
    return model.eval()

def get_batch(corpus: EncodedSRLCorpus, idx: int, device: torch.device) -> Dict[str, Tensor]:
    x = corpus.encoded[idx]
    return {'input_ids': x.input_ids.unsqueeze(0).to(device), 'attention_mask': x.attention_mask.unsqueeze(0).to(device), 'labels': x.labels.unsqueeze(0).to(device)}

@torch.no_grad()
def extract_token_records(split: str, corpus: EncodedSRLCorpus, model: torch.nn.Module, device: torch.device, id2label: Dict[int, str], layer: int) -> List[TokenRecord]:
    records: List[TokenRecord] = []
    model.eval()
    for inst_idx in tqdm(range(len(corpus)), desc=f'Extracting {split} records'):
        batch = get_batch(corpus, inst_idx, device)
        out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True, return_dict=True)
        if layer >= len(out.hidden_states):
            raise ValueError(f'Layer {layer} unavailable; model returned {len(out.hidden_states)} hidden-state tensors.')
        hidden = out.hidden_states[layer][0]
        logits = out.logits[0]
        labels = batch['labels'][0]
        inst = corpus.instances[inst_idx]
        for word_idx, tok_pos in corpus.first_subtoken_pos[inst_idx].items():
            label_id = int(labels[tok_pos].item())
            if label_id == -100:
                continue
            loss = F.cross_entropy(logits[tok_pos].view(1, -1), torch.tensor([label_id], device=device))
            pred_id = int(torch.argmax(logits[tok_pos]).item())
            records.append(TokenRecord(split, inst_idx, word_idx, inst.tokens[word_idx], id2label[label_id], label_id, hidden[tok_pos].detach().cpu().float().numpy(), float(loss.detach().cpu()), pred_id, pred_id == label_id))
    return records

def l2_normalise(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)

def compute_lcts(train_records: List[TokenRecord], test_records: List[TokenRecord]) -> Tuple[np.ndarray, np.ndarray]:
    train_h = np.stack([r.hidden for r in train_records]).astype(np.float32)
    test_h = np.stack([r.hidden for r in test_records]).astype(np.float32)
    dim = train_h.shape[1]
    by_label = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label[r.label].append(i)
    stats = {}
    for lab, idxs in by_label.items():
        H = train_h[idxs]
        c = H.mean(axis=0)
        res = H - c
        mu, std = (res.mean(axis=0), res.std(axis=0))
        std = np.where(std < 1e-06, 1.0, std)
        stats[lab] = (c, mu, std)
    gc, gmu, gstd = (train_h.mean(axis=0), np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))

    def tr(records: List[TokenRecord], H: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(H)
        for i, r in enumerate(records):
            c, mu, std = stats.get(r.label, (gc, gmu, gstd))
            Z[i] = (H[i] - c - mu) / std
        return Z
    return (tr(train_records, train_h), tr(test_records, test_h))

def retrieve_lcts(train_records: List[TokenRecord], test_records: List[TokenRecord], train_z: np.ndarray, test_z: np.ndarray, top_k: int) -> Dict[int, List[int]]:
    by_label = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label[r.label].append(i)
    ret: Dict[int, List[int]] = {}
    for ti, r in enumerate(tqdm(test_records, desc='Retrieving LCTS candidates')):
        idxs = by_label.get(r.label, [])
        if not idxs:
            ret[ti] = []
            continue
        d = np.linalg.norm(train_z[idxs] - test_z[ti], axis=1)
        ret[ti] = [idxs[int(j)] for j in np.argsort(d)[:top_k]]
    return ret

def retrieve_raw(train_records: List[TokenRecord], test_records: List[TokenRecord], top_k: int) -> Dict[int, List[int]]:
    train_h = l2_normalise(np.stack([r.hidden for r in train_records]).astype(np.float32))
    test_h = l2_normalise(np.stack([r.hidden for r in test_records]).astype(np.float32))
    by_label = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label[r.label].append(i)
    ret: Dict[int, List[int]] = {}
    for ti, r in enumerate(tqdm(test_records, desc='Retrieving raw candidates')):
        idxs = by_label.get(r.label, [])
        if not idxs:
            ret[ti] = []
            continue
        sim = train_h[idxs] @ test_h[ti]
        ret[ti] = [idxs[int(j)] for j in np.argsort(-sim)[:top_k]]
    return ret

def select_gradient_params(model: torch.nn.Module, mode: str) -> List[torch.nn.Parameter]:
    named = list(model.named_parameters())
    if mode == 'classifier':
        params = [p for n, p in named if p.requires_grad and ('classifier' in n or 'score' in n)]
    elif mode == 'encoder_last':
        params = [p for n, p in named if p.requires_grad and ('classifier' in n or 'score' in n or 'encoder.layer.11' in n)]
    elif mode == 'all':
        params = [p for _, p in named if p.requires_grad]
    else:
        raise ValueError(mode)
    if not params:
        raise ValueError('No gradient parameters selected. Try --gradient_params all.')
    return params

class GradientComputer:

    def __init__(self, model: torch.nn.Module, train_corpus: EncodedSRLCorpus, test_corpus: EncodedSRLCorpus, device: torch.device, params: List[torch.nn.Parameter]):
        self.model = model
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.device = device
        self.params = params
        self.cache: Dict[str, Tensor] = {}

    def gradient(self, rec: TokenRecord) -> Tensor:
        if rec.uid in self.cache:
            return self.cache[rec.uid]
        corpus = self.train_corpus if rec.split == 'train' else self.test_corpus
        tok_pos = corpus.first_subtoken_pos[rec.instance_idx][rec.word_idx]
        batch = get_batch(corpus, rec.instance_idx, self.device)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_dict=True)
        loss = F.cross_entropy(out.logits[0, tok_pos].view(1, -1), torch.tensor([rec.label_id], device=self.device))
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, create_graph=False, allow_unused=True)
        flat = torch.cat([torch.zeros(p.numel()) if g is None else g.detach().cpu().float().flatten() for g, p in zip(grads, self.params)])
        self.cache[rec.uid] = flat
        return flat

def safe_cosine(a: Tensor, b: Tensor, eps: float=1e-12) -> float:
    denom = torch.norm(a) * torch.norm(b)
    return 0.0 if float(denom) < eps else float(torch.dot(a, b) / denom)

def pair_score(train_rec: TokenRecord, g_train: Tensor, g_test: Tensor, method: str) -> float:
    dot = float(torch.dot(g_train, g_test))
    if method == 'ifstar':
        return math.exp(-train_rec.loss) * safe_cosine(g_train, g_test)
    if method in {'tracin_dot', 'hf_classical_if'}:
        return dot
    raise ValueError(method)

def evaluate(name: str, train_records: List[TokenRecord], test_records: List[TokenRecord], retrieval: Dict[int, List[int]], grad_comp: GradientComputer, method: str, top_k: int, random_n: int, seed: int, max_test_tokens: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label[r.label].append(i)
    indices = list(range(len(test_records)))[:max_test_tokens] if max_test_tokens > 0 else list(range(len(test_records)))
    iss_values, ipc_hits, ipc_total, details = ([], 0, 0, [])
    for ti in tqdm(indices, desc=f'Evaluating {name}'):
        test_rec = test_records[ti]
        cand = retrieval.get(ti, [])[:top_k]
        pool = by_label.get(test_rec.label, [])
        if not cand or not pool:
            continue
        rand = pool if len(pool) <= random_n else rng.sample(pool, random_n)
        g_test = grad_comp.gradient(test_rec)
        cand_scores = [pair_score(train_records[ci], grad_comp.gradient(train_records[ci]), g_test, method) for ci in cand]
        rand_scores = [pair_score(train_records[ri], grad_comp.gradient(train_records[ri]), g_test, method) for ri in rand]
        iss = float(np.mean([100.0 * sum((1 for rs in rand_scores if rs < cs)) / len(rand_scores) for cs in cand_scores]))
        iss_values.append(iss)
        pred_support = float(np.mean(cand_scores)) > 0
        if test_rec.correct and pred_support or (not test_rec.correct and (not pred_support)):
            ipc_hits += 1
        ipc_total += 1
        details.append({'configuration': name, 'test_uid': test_rec.uid, 'test_token': test_rec.token, 'test_label': test_rec.label, 'test_correct': int(test_rec.correct), 'mean_score': float(np.mean(cand_scores)), 'iss': iss, 'retrieved_train_uids': ' '.join((train_records[i].uid for i in cand)), 'retrieved_train_tokens': ' '.join((train_records[i].token for i in cand))})
    return ({'configuration': name, 'ISS': float(np.mean(iss_values)) if iss_values else float('nan'), 'IPC': 100.0 * ipc_hits / ipc_total if ipc_total else float('nan'), 'num_test_tokens': ipc_total}, details)

def write_outputs(output_dir: Path, rows: List[Dict[str, Any]], details: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'experiment_d_ontonotes_table5.csv'
    md_path = output_dir / 'experiment_d_ontonotes_table5.md'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['configuration', 'ISS', 'IPC', 'num_test_tokens'])
        w.writeheader()
        for r in rows:
            w.writerow({'configuration': r['configuration'], 'ISS': f"{r['ISS']:.2f}", 'IPC': f"{r['IPC']:.2f}", 'num_test_tokens': r['num_test_tokens']})
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('Table 5. General-domain validation on OntoNotes v5.0 at layer 10.\n\n')
        f.write('| Configuration | ISS | IPC | # test tokens |\n|---|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['configuration']} | {r['ISS']:.2f} | {r['IPC']:.2f} | {r['num_test_tokens']} |\n")
    if details:
        dpath = output_dir / 'experiment_d_token_details.csv'
        with open(dpath, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
            w.writeheader()
            w.writerows(details)
    print(f'Saved {csv_path}')
    print(f'Saved {md_path}')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run.')
    p.add_argument('--source', choices=['hf', 'conll'], default='hf')
    p.add_argument('--hf_dataset', default='conll2012_ontonotesv5')
    p.add_argument('--hf_config', default=None)
    p.add_argument('--hf_train_split', default='train')
    p.add_argument('--hf_test_split', default='test')
    p.add_argument('--train_file', default=None)
    p.add_argument('--test_file', default=None)
    p.add_argument('--model_checkpoint', default='bert-base-cased')
    p.add_argument('--output_dir', default='outputs/experiment_d_ontonotes')
    p.add_argument('--do_train', action='store_true')
    p.add_argument('--train_epochs', type=int, default=2)
    p.add_argument('--train_batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=2e-05)
    p.add_argument('--layer', type=int, default=10)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--random_n', type=int, default=100)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--gradient_params', choices=['classifier', 'encoder_last', 'all'], default='classifier')
    p.add_argument('--max_train_instances', type=int, default=0)
    p.add_argument('--max_test_instances', type=int, default=0)
    p.add_argument('--max_test_tokens', type=int, default=0)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    print('Loading OntoNotes / CoNLL-2012 SRL instances...')
    if args.source == 'hf':
        train_instances = load_hf_ontonotes(args.hf_dataset, args.hf_config, args.hf_train_split)
        test_instances = load_hf_ontonotes(args.hf_dataset, args.hf_config, args.hf_test_split)
    else:
        if not args.train_file or not args.test_file:
            raise ValueError('--train_file and --test_file are required for --source conll')
        train_instances = read_local_conll2012(args.train_file)
        test_instances = read_local_conll2012(args.test_file)
    if args.max_train_instances > 0:
        train_instances = train_instances[:args.max_train_instances]
    if args.max_test_instances > 0:
        test_instances = test_instances[:args.max_test_instances]
    print(f'Train SRL instances: {len(train_instances)}')
    print(f'Test SRL instances : {len(test_instances)}')
    if not train_instances or not test_instances:
        raise RuntimeError('No SRL instances loaded.')
    label2id, id2label = build_label_maps(train_instances, test_instances)
    print(f'Number of SRL labels: {len(label2id)}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    train_corpus = EncodedSRLCorpus(train_instances, tokenizer, label2id, args.max_length)
    test_corpus = EncodedSRLCorpus(test_instances, tokenizer, label2id, args.max_length)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if args.do_train:
        model = train_token_classifier(model, train_corpus, tokenizer, output_dir, device, args.train_epochs, args.train_batch_size, args.learning_rate)
    else:
        print('Skipping training. Use a fine-tuned checkpoint or pass --do_train.')
    train_records = extract_token_records('train', train_corpus, model, device, id2label, args.layer)
    test_records = extract_token_records('test', test_corpus, model, device, id2label, args.layer)
    print(f'Train token records: {len(train_records)}')
    print(f'Test token records : {len(test_records)}')
    train_z, test_z = compute_lcts(train_records, test_records)
    lcts_retrieval = retrieve_lcts(train_records, test_records, train_z, test_z, args.top_k)
    raw_retrieval = retrieve_raw(train_records, test_records, args.top_k)
    params = select_gradient_params(model, args.gradient_params)
    print(f'Gradient parameter mode: {args.gradient_params}; selected params={sum((p.numel() for p in params)):,}')
    grad_comp = GradientComputer(model, train_corpus, test_corpus, device, params)
    configs = [('LCTS + IF* (full)', lcts_retrieval, 'ifstar'), ('LCTS + TracIn-style gradient-dot-product score', lcts_retrieval, 'tracin_dot'), ('LCTS + Hessian-free classical-IF approximation', lcts_retrieval, 'hf_classical_if'), ('Raw representations + TracIn-style gradient-dot-product score', raw_retrieval, 'tracin_dot')]
    rows, all_details = ([], [])
    for name, retrieval, method in configs:
        row, details = evaluate(name, train_records, test_records, retrieval, grad_comp, method, args.top_k, args.random_n, args.seed, args.max_test_tokens)
        rows.append(row)
        all_details.extend(details)
    write_outputs(output_dir, rows, all_details)
    print('Done. Done.')
if __name__ == '__main__':
    main()
