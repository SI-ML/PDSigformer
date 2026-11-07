"""固定 split42 缓存的可移植加载器；不重新划分、标准化或改变 padding。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import collate_fn


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class WearableLoader(Dataset):
    def __init__(self, root_path, args=None, flag="test", expected_files=None):
        self.root = Path(root_path)
        split = {"val": "valid", "vali": "valid"}.get(flag.lower(), flag.lower())
        if split not in ("train", "valid", "test"):
            raise ValueError("分区必须为 train/valid/test")
        self.split = split
        metadata_name = "manifest.json" if (self.root / "manifest.json").exists() else "task.json"
        metadata_sha = (expected_files or {}).get(metadata_name)
        if metadata_sha and file_sha(self.root / metadata_name) != metadata_sha:
            raise ValueError("数据清单 SHA256 不符")
        if (self.root / "manifest.json").exists():
            meta = json.loads((self.root / "manifest.json").read_text(encoding="utf-8-sig"))
            if (
                meta["split_seed"] != 42
                or meta["normalization"]["method"] != "StandardScaler_per_window_per_channel"
            ):
                raise ValueError("缓存协议不匹配")
            info = meta["splits"][split]
            xp, yp, ip = (self.root / info[k] for k in ("X_file", "y_file", "mapping_file"))
            expected = {str(self.root / k): v for k, v in meta["artifacts"].items()}
        else:
            meta = json.loads((self.root / "task.json").read_text(encoding="utf-8-sig"))
            xp, yp, ip = (self.root / split / n for n in ("X.npy", "y.npy", "identities.csv"))
            expected = {}
        for p in (xp, yp, ip):
            if not p.is_file():
                raise FileNotFoundError(f"缺少 {split} 数据：{p}")
            wanted = expected.get(str(p)) or (expected_files or {}).get(
                p.relative_to(self.root).as_posix()
            )
            if wanted and file_sha(p) != wanted:
                raise ValueError(f"数据 SHA256 不匹配：{p}")
        self.X = np.load(xp, allow_pickle=False)
        self.y = np.load(yp, allow_pickle=False)
        with ip.open(encoding="utf-8-sig", newline="") as f:
            self.window_records = list(csv.DictReader(f))
        if self.X.ndim != 3 or self.X.dtype != np.float32 or not np.isfinite(self.X).all():
            raise ValueError("X 必须为有限 float32 [N,T,C]")
        if self.y.shape != (len(self.X),) or self.y.dtype != np.int64:
            raise ValueError("y 必须为 int64 [N]")
        if len(self.window_records) != len(self.X):
            raise ValueError("身份记录数量不符")
        for i, row in enumerate(self.window_records):
            if int(row["label"]) != int(self.y[i]) or int(row["loader_index"]) != i:
                raise ValueError("样本身份/顺序/标签不符")
        self.subject_ids = [r["subject_id"] for r in self.window_records]
        self.window_ids = [r["window_id"] for r in self.window_records]
        self.max_seq_len, self.num_channels = self.X.shape[1:]
        self.class_names = ["HC", "PD"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.as_tensor(self.y[index], dtype=torch.long)


def make_loader(dataset, batch_size, seed, train=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=0,
        generator=generator,
        collate_fn=lambda data: collate_fn(data, max_len=dataset.max_seq_len),
    )
