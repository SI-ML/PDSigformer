"""Configuration, random state, metrics, and strict checkpoint serialization."""
import csv
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def metrics(y, prob):
    pred = prob.argmax(1)
    onehot = np.eye(2)[y]
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_precision=float(precision_score(y, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y, pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        macro_auroc=float(roc_auc_score(onehot, prob, average="macro")),
        macro_auprc=float(average_precision_score(onehot, prob, average="macro")),
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_trusted(path, expected_sha=None):
    # Load only the user's own training outputs or historical checkpoints verified against the SHA manifest.
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError("checkpoint SHA256 不符")
    return torch.load(path, map_location="cpu", weights_only=False)


def state_hash(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_predictions(path, dataset, prob):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "window_id", "subject_id", "y_true", "p0", "p1", "y_pred"])
        for i, (y, p) in enumerate(zip(dataset.y, prob)):
            w.writerow(
                [
                    i,
                    dataset.window_ids[i],
                    dataset.subject_ids[i],
                    int(y),
                    float(p[0]),
                    float(p[1]),
                    int(p.argmax()),
                ]
            )
