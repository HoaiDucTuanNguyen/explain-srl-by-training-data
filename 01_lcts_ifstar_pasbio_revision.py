from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except Exception as exc:
    raise RuntimeError('This script requires transformers. Install with: pip install transformers') from exc

@dataclass
class SentenceExample:
    tokens: List[str]
    labels: List[str]
    sent_id: str = ''

@dataclass
class TokenRecord:
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

    @property
    def uid(self) -> str:
        return f'{self.split}:{self.sent_idx}:{self.word_idx}'

def read_pasbio_jsonl(path: str) -> List[SentenceExample]:
    examples: List[SentenceExample] = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj.get('tokens')
            labels = obj.get('labels') or obj.get('srl_labels')
            if tokens is None or labels is None:
                raise ValueError(f"JSONL line {i + 1} must contain 'tokens' and 'labels' fields.")
            if len(tokens) != len(labels):
                raise ValueError(f'JSONL line {i + 1}: tokens and labels have different lengths.')
            examples.append(SentenceExample(tokens=list(tokens), labels=list(labels), sent_id=str(obj.get('sent_id', i))))
    return examples

def read_pasbio_conll(path: str, token_col: int=0, label_col: int=-1, delimiter: Optional[str]=None) -> List[SentenceExample]:
    examples: List[SentenceExample] = []
    tokens: List[str] = []
    labels: List[str] = []
    sent_id = 0

    def flush():
        nonlocal sent_id, tokens, labels
        if tokens:
            if len(tokens) != len(labels):
                raise ValueError(f'Sentence {sent_id}: token/label length mismatch.')
            examples.append(SentenceExample(tokens=tokens, labels=labels, sent_id=str(sent_id)))
            sent_id += 1
            tokens, labels = ([], [])
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith('#'):
                continue
            parts = line.split(delimiter)
            if len(parts) == 1:
                parts = line.split()
            try:
                token = parts[token_col]
                label = parts[label_col]
            except IndexError as exc:
                raise ValueError(f'Line {line_no}: cannot read token_col={token_col}, label_col={label_col}. Line content: {line}') from exc
            tokens.append(token)
            labels.append(label)
    flush()
    return examples

def load_pasbio(path: str, fmt: str, token_col: int, label_col: int) -> List[SentenceExample]:
    if fmt == 'jsonl':
        return read_pasbio_jsonl(path)
    if fmt == 'conll':
        return read_pasbio_conll(path, token_col=token_col, label_col=label_col)
    raise ValueError(f'Unsupported format: {fmt}')

class EncodedCorpus:

    def __init__(self, examples: List[SentenceExample], tokenizer: Any, label2id: Dict[str, int], max_length: int=256) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.encodings: List[Dict[str, Tensor]] = []
        self.word_id_lists: List[List[Optional[int]]] = []
        self.first_subtoken_pos: List[Dict[int, int]] = []
        self._encode()

    def _encode(self) -> None:
        for ex in self.examples:
            enc = self.tokenizer(ex.tokens, is_split_into_words=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            word_ids = enc.word_ids(batch_index=0)
            label_ids = []
            first_pos: Dict[int, int] = {}
            seen = set()
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    label_ids.append(-100)
                elif wid not in seen:
                    seen.add(wid)
                    first_pos[wid] = pos
                    if wid >= len(ex.labels):
                        label_ids.append(-100)
                    else:
                        label_ids.append(self.label2id[ex.labels[wid]])
                else:
                    label_ids.append(-100)
            enc['labels'] = torch.tensor([label_ids], dtype=torch.long)
            self.encodings.append(enc)
            self.word_id_lists.append(word_ids)
            self.first_subtoken_pos.append(first_pos)

    def __len__(self) -> int:
        return len(self.examples)

def build_label_mapping(*corpora: List[SentenceExample]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({lab for corpus in corpora for ex in corpus for lab in ex.labels})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return (label2id, id2label)

def move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {k: v.to(device) for k, v in batch.items()}

def select_gradient_parameters(model: torch.nn.Module, mode: str) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    named = list(model.named_parameters())
    if mode == 'classifier':
        for name, p in named:
            if p.requires_grad and ('classifier' in name or 'score' in name or 'crf' in name):
                params.append(p)
    elif mode == 'encoder_last':
        for name, p in named:
            if not p.requires_grad:
                continue
            if 'classifier' in name or 'score' in name:
                params.append(p)
            if 'encoder.layer.11' in name or 'bert.encoder.layer.11' in name:
                params.append(p)
    elif mode == 'all':
        params = [p for _, p in named if p.requires_grad]
    else:
        raise ValueError(f'Unknown gradient_params mode: {mode}')
    if not params:
        raise ValueError(f'No parameters selected for mode={mode}. Try --gradient_params all.')
    return params

@torch.no_grad()
def extract_token_records(split: str, corpus: EncodedCorpus, model: torch.nn.Module, device: torch.device, id2label: Dict[int, str], layer: int) -> List[TokenRecord]:
    model.eval()
    records: List[TokenRecord] = []
    for sent_idx in tqdm(range(len(corpus)), desc=f'Extracting {split} representations'):
        batch = move_batch_to_device(corpus.encodings[sent_idx], device)
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'), token_type_ids=batch.get('token_type_ids', None), labels=None, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0]
        hidden_states = outputs.hidden_states
        if layer < 0:
            layer_idx = layer
        else:
            layer_idx = layer
        if abs(layer_idx) >= len(hidden_states):
            raise ValueError(f'Requested layer={layer}, but model returned {len(hidden_states)} hidden-state tensors.')
        selected_hidden = hidden_states[layer_idx][0]
        labels = batch['labels'][0]
        for word_idx, pos in corpus.first_subtoken_pos[sent_idx].items():
            if word_idx >= len(corpus.examples[sent_idx].tokens):
                continue
            label_id = int(labels[pos].item())
            if label_id == -100:
                continue
            logit = logits[pos]
            loss = F.cross_entropy(logit.view(1, -1), torch.tensor([label_id], device=device))
            pred_id = int(torch.argmax(logit).item())
            token = corpus.examples[sent_idx].tokens[word_idx]
            label = id2label[label_id]
            records.append(TokenRecord(split=split, sent_idx=sent_idx, word_idx=word_idx, token=token, label=label, label_id=label_id, hidden=selected_hidden[pos].detach().cpu().float().numpy(), loss=float(loss.detach().cpu().item()), pred_id=pred_id, correct=pred_id == label_id))
    return records

def l2_normalize(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)

def cosine_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_n = l2_normalize(x)
    y_n = l2_normalize(y)
    return 1.0 - np.matmul(x_n, y_n.T)

def compute_lcts_vectors(train_records: List[TokenRecord], test_records: List[TokenRecord], eps: float=1e-06) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    train_h = np.stack([r.hidden for r in train_records])
    test_h = np.stack([r.hidden for r in test_records])
    dim = train_h.shape[1]
    by_label_train = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label_train[r.label].append(i)
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for label, idxs in by_label_train.items():
        H = train_h[idxs]
        centroid = H.mean(axis=0)
        residuals = H - centroid
        mu = residuals.mean(axis=0)
        std = residuals.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        stats[label] = {'centroid': centroid, 'mu': mu, 'std': std}
    global_centroid = train_h.mean(axis=0)
    global_mu = np.zeros(dim)
    global_std = np.ones(dim)

    def transform(records: List[TokenRecord], H_all: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(H_all, dtype=np.float32)
        for i, r in enumerate(records):
            st = stats.get(r.label, {'centroid': global_centroid, 'mu': global_mu, 'std': global_std})
            residual = H_all[i] - st['centroid']
            Z[i] = ((residual - st['mu']) / st['std']).astype(np.float32)
        return Z
    train_z = transform(train_records, train_h)
    test_z = transform(test_records, test_h)
    cov_inv_by_label: Dict[str, np.ndarray] = {}
    for label, idxs in by_label_train.items():
        Z = train_z[idxs]
        if Z.shape[0] < 3:
            cov_inv_by_label[label] = np.eye(dim, dtype=np.float32)
            continue
        cov = np.cov(Z, rowvar=False)
        alpha = 0.001
        cov = (1 - alpha) * cov + alpha * np.eye(dim)
        cov_inv_by_label[label] = np.linalg.pinv(cov).astype(np.float32)
    return (train_z, test_z, cov_inv_by_label)

def retrieve_lcts_topk(train_records: List[TokenRecord], test_records: List[TokenRecord], train_z: np.ndarray, test_z: np.ndarray, cov_inv_by_label: Dict[str, np.ndarray], top_k: int, mahalanobis: str='diag') -> Dict[int, List[int]]:
    by_label_train = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label_train[r.label].append(i)
    result: Dict[int, List[int]] = {}
    for ti, tr in enumerate(tqdm(test_records, desc='Retrieving LCTS top-k')):
        idxs = by_label_train.get(tr.label, [])
        if not idxs:
            result[ti] = []
            continue
        Z = train_z[idxs]
        diff = Z - test_z[ti]
        if mahalanobis == 'full':
            inv_cov = cov_inv_by_label.get(tr.label)
            if inv_cov is None:
                d = np.linalg.norm(diff, axis=1)
            else:
                d = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, inv_cov, diff), 0.0))
        elif mahalanobis == 'diag':
            d = np.linalg.norm(diff, axis=1)
        else:
            raise ValueError("mahalanobis must be 'diag' or 'full'.")
        order = np.argsort(d)[:top_k]
        result[ti] = [idxs[int(j)] for j in order]
    return result

def retrieve_raw_cosine_topk(train_records: List[TokenRecord], test_records: List[TokenRecord], top_k: int) -> Dict[int, List[int]]:
    train_h = np.stack([r.hidden for r in train_records])
    test_h = np.stack([r.hidden for r in test_records])
    by_label_train = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label_train[r.label].append(i)
    result: Dict[int, List[int]] = {}
    for ti, tr in enumerate(tqdm(test_records, desc='Retrieving raw-cosine top-k')):
        idxs = by_label_train.get(tr.label, [])
        if not idxs:
            result[ti] = []
            continue
        d = cosine_distance_matrix(test_h[ti:ti + 1], train_h[idxs])[0]
        order = np.argsort(d)[:top_k]
        result[ti] = [idxs[int(j)] for j in order]
    return result

class GradientComputer:

    def __init__(self, model: torch.nn.Module, train_corpus: EncodedCorpus, test_corpus: EncodedCorpus, device: torch.device, params: List[torch.nn.Parameter]) -> None:
        self.model = model
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.device = device
        self.params = params
        self.cache: Dict[str, Tensor] = {}

    def _corpus_for_split(self, split: str) -> EncodedCorpus:
        if split == 'train':
            return self.train_corpus
        if split == 'test':
            return self.test_corpus
        raise ValueError(f'Unknown split: {split}')

    def gradient(self, rec: TokenRecord) -> Tensor:
        if rec.uid in self.cache:
            return self.cache[rec.uid]
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        corpus = self._corpus_for_split(rec.split)
        batch = move_batch_to_device(corpus.encodings[rec.sent_idx], self.device)
        pos = corpus.first_subtoken_pos[rec.sent_idx].get(rec.word_idx)
        if pos is None:
            raise ValueError(f'Cannot locate token {rec.uid} in tokenized sequence.')
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'), token_type_ids=batch.get('token_type_ids', None), labels=None, return_dict=True)
        logits = outputs.logits[0, pos]
        target = torch.tensor([rec.label_id], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits.view(1, -1), target)
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, create_graph=False, allow_unused=True)
        flat_parts = []
        for g, p in zip(grads, self.params):
            if g is None:
                flat_parts.append(torch.zeros_like(p, device='cpu').flatten())
            else:
                flat_parts.append(g.detach().cpu().float().flatten())
        flat = torch.cat(flat_parts)
        self.cache[rec.uid] = flat
        return flat

def safe_cosine(a: Tensor, b: Tensor, eps: float=1e-12) -> float:
    denom = torch.norm(a) * torch.norm(b)
    if float(denom) < eps:
        return 0.0
    return float(torch.dot(a, b) / denom)

def score_pair(train_rec: TokenRecord, test_rec: TokenRecord, grad_train: Tensor, grad_test: Tensor, scoring: str) -> float:
    dot = float(torch.dot(grad_train, grad_test))
    if scoring == 'ifstar':
        return math.exp(-float(train_rec.loss)) * safe_cosine(grad_train, grad_test)
    if scoring == 'tracin_dot':
        return dot
    if scoring == 'hf_classical_if_support':
        return dot
    raise ValueError(f'Unknown scoring mode: {scoring}')

def evaluate_configuration(name: str, train_records: List[TokenRecord], test_records: List[TokenRecord], retrieval: Dict[int, List[int]], grad_computer: GradientComputer, scoring: str, top_k: int, random_n: int, seed: int=13, max_test_tokens: Optional[int]=None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label_train = defaultdict(list)
    for i, r in enumerate(train_records):
        by_label_train[r.label].append(i)
    iss_values: List[float] = []
    ipc_hits = 0
    ipc_total = 0
    details: List[Dict[str, Any]] = []
    test_indices = list(range(len(test_records)))
    if max_test_tokens is not None and max_test_tokens > 0:
        test_indices = test_indices[:max_test_tokens]
    for ti in tqdm(test_indices, desc=f'Evaluating {name}'):
        test_rec = test_records[ti]
        candidate_idxs = retrieval.get(ti, [])[:top_k]
        if not candidate_idxs:
            continue
        same_label_pool = by_label_train.get(test_rec.label, [])
        if not same_label_pool:
            continue
        if len(same_label_pool) <= random_n:
            random_idxs = same_label_pool[:]
        else:
            random_idxs = rng.sample(same_label_pool, random_n)
        g_test = grad_computer.gradient(test_rec)
        candidate_scores: List[float] = []
        for ci in candidate_idxs:
            train_rec = train_records[ci]
            g_train = grad_computer.gradient(train_rec)
            s = score_pair(train_rec, test_rec, g_train, g_test, scoring=scoring)
            candidate_scores.append(s)
        random_scores: List[float] = []
        for ri in random_idxs:
            train_rec = train_records[ri]
            g_train = grad_computer.gradient(train_rec)
            s = score_pair(train_rec, test_rec, g_train, g_test, scoring=scoring)
            random_scores.append(s)
        if not random_scores:
            continue
        candidate_iss = []
        for s in candidate_scores:
            count = sum((1 for rs in random_scores if rs < s))
            candidate_iss.append(100.0 * count / len(random_scores))
        iss = float(np.mean(candidate_iss))
        iss_values.append(iss)
        mean_score = float(np.mean(candidate_scores))
        predicted_supportive = mean_score > 0
        if test_rec.correct and predicted_supportive or (not test_rec.correct and (not predicted_supportive)):
            ipc_hits += 1
        ipc_total += 1
        details.append({'configuration': name, 'test_uid': test_rec.uid, 'test_token': test_rec.token, 'test_label': test_rec.label, 'test_correct': int(test_rec.correct), 'mean_score': mean_score, 'iss': iss, 'retrieved_train_uids': ' '.join((train_records[i].uid for i in candidate_idxs)), 'retrieved_train_tokens': ' '.join((train_records[i].token for i in candidate_idxs))})
    metrics = {'configuration': name, 'ISS': float(np.mean(iss_values)) if iss_values else float('nan'), 'IPC': 100.0 * ipc_hits / ipc_total if ipc_total else float('nan'), 'num_test_tokens': int(ipc_total)}
    return (metrics, details)

def write_results(output_dir: Path, rows: List[Dict[str, float]], details: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'results.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['configuration', 'ISS', 'IPC', 'num_test_tokens'])
        writer.writeheader()
        for r in rows:
            writer.writerow({'configuration': r['configuration'], 'ISS': f"{r['ISS']:.2f}", 'IPC': f"{r['IPC']:.2f}", 'num_test_tokens': r['num_test_tokens']})
    md_path = output_dir / 'results.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| Configuration | ISS | IPC | # test tokens |\n')
        f.write('|---|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['configuration']} | {r['ISS']:.2f} | {r['IPC']:.2f} | {r['num_test_tokens']} |\n")
    details_path = output_dir / 'details.csv'
    if details:
        with open(details_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = list(details[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in details:
                writer.writerow(d)
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {md_path}')
    if details:
        print(f'Saved: {details_path}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_file', required=True, help='')
    parser.add_argument('--test_file', required=True, help='')
    parser.add_argument('--format', choices=['conll', 'jsonl'], default='conll', help='')
    parser.add_argument('--token_col', type=int, default=0, help='')
    parser.add_argument('--label_col', type=int, default=-1, help='')
    parser.add_argument('--model_checkpoint', required=True, help='')
    parser.add_argument('--output_dir', default='outputs/reviewer_baselines')
    parser.add_argument('--layer', type=int, default=10, help="")
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--random_n', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--gradient_params', choices=['classifier', 'encoder_last', 'all'], default='classifier', help="")
    parser.add_argument('--mahalanobis', choices=['diag', 'full'], default='diag', help="")
    parser.add_argument('--max_test_tokens', type=int, default=0, help='')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    print('loading')
    train_examples = load_pasbio(args.train_file, args.format, args.token_col, args.label_col)
    test_examples = load_pasbio(args.test_file, args.format, args.token_col, args.label_col)
    print(f'Train sentences: {len(train_examples)}')
    print(f'Test sentences : {len(test_examples)}')
    label2id, id2label = build_label_mapping(train_examples, test_examples)
    print(f'Labels: {len(label2id)}')
    print('Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()
    print('encoding')
    train_corpus = EncodedCorpus(train_examples, tokenizer, label2id, max_length=args.max_length)
    test_corpus = EncodedCorpus(test_examples, tokenizer, label2id, max_length=args.max_length)
    print('extracting')
    train_records = extract_token_records('train', train_corpus, model, device, id2label, args.layer)
    test_records = extract_token_records('test', test_corpus, model, device, id2label, args.layer)
    print(f'Train tokens: {len(train_records)}')
    print(f'Test tokens : {len(test_records)}')
    print('processing')
    train_z, test_z, cov_inv_by_label = compute_lcts_vectors(train_records, test_records)
    print('processing')
    lcts_retrieval = retrieve_lcts_topk(train_records, test_records, train_z, test_z, cov_inv_by_label, top_k=args.top_k, mahalanobis=args.mahalanobis)
    raw_retrieval = retrieve_raw_cosine_topk(train_records, test_records, top_k=args.top_k)
    print('processing')
    params = select_gradient_parameters(model, args.gradient_params)
    num_params = sum((p.numel() for p in params))
    print(f'Gradient parameter mode: {args.gradient_params}')
    print(f'Number of selected gradient parameters: {num_params:,}')
    grad_computer = GradientComputer(model, train_corpus, test_corpus, device, params)
    max_test_tokens = args.max_test_tokens if args.max_test_tokens > 0 else None
    configs = [{'name': 'm0', 'retrieval': lcts_retrieval, 'scoring': 'ifstar'}, {'name': 'm1', 'retrieval': lcts_retrieval, 'scoring': 'tracin_dot'}, {'name': 'm2', 'retrieval': lcts_retrieval, 'scoring': 'hf_classical_if_support'}, {'name': 'm3', 'retrieval': raw_retrieval, 'scoring': 'tracin_dot'}]
    all_rows: List[Dict[str, float]] = []
    all_details: List[Dict[str, Any]] = []
    for cfg in configs:
        metrics, details = evaluate_configuration(name=cfg['name'], train_records=train_records, test_records=test_records, retrieval=cfg['retrieval'], grad_computer=grad_computer, scoring=cfg['scoring'], top_k=args.top_k, random_n=args.random_n, seed=args.seed, max_test_tokens=max_test_tokens)
        all_rows.append(metrics)
        all_details.extend(details)
    write_results(output_dir, all_rows, all_details)
    print('\nDone.')
if __name__ == '__main__':
    main()
