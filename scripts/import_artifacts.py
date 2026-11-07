"""Import required archived artifacts and verify the provenance of recovered training arrays."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    a = p.parse_args()
    src, dst = a.source.resolve(), a.destination.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("来源和目标须为独立目录")
    dst.mkdir(parents=True, exist_ok=True)
    required, restored = {}, []
    for name in ("baseline78_unified_v4", "pds24"):
        rel = "manifests/" + name + ".json"
        required[rel] = sha(src / rel)
        for r in json.loads((src / rel).read_text())["runs"]:
            for key in ("data_files", "evidence_files", "runtime_files"):
                required.update(r.get(key, {}))
            cp = r.get("checkpoint", r.get("checkpoint_relative_path"))
            required[cp] = r["sha"]
            for key in (
                "expected_predictions",
                "effective_config",
                "selection_record",
                "original_test_summary",
            ):
                if r.get(key):
                    required[r[key]] = r.get(key + "_sha256") or sha(src / r[key])
    # Recover missing PDS Train/Valid arrays only from baseline arrays with identical archived hashes.
    for dataset, task in (
        ("shimmer", "Shimmer_11_session11_DRINK"),
        ("pads", "PADS_10_task07_CrossArms"),
    ):
        base = "data_snapshots/pds_" + dataset
        m = json.loads((src / base / "manifest.json").read_text())
        required[base + "/manifest.json"] = sha(src / base / "manifest.json")
        for rel, digest in m["artifacts"].items():
            required[base + "/" + rel] = digest
        for split in ("train", "valid", "test"):
            pd = src / base / m["splits"][split]["mapping_file"]
            bd = src / "data_snapshots/baseline_common/prepared" / task / split / "identities.csv"
            with pd.open(encoding="utf-8-sig", newline="") as f:
                pr = list(csv.DictReader(f))
            with bd.open(encoding="utf-8-sig", newline="") as f:
                br = list(csv.DictReader(f))
            common = sorted(set(pr[0]) & set(br[0]))
            if len(pr) != len(br) or any(any(x[k] != y[k] for k in common) for x, y in zip(pr, br)):
                raise ValueError("恢复来源的样本身份或顺序不一致: " + dataset + "/" + split)
            for field, fname in (("X_file", "X.npy"), ("y_file", "y.npy")):
                target = base + "/" + m["splits"][split][field]
                source = src / target
                if not source.exists():
                    source = src / "data_snapshots/baseline_common/prepared" / task / split / fname
                    digest = required[target]
                    if sha(source) != digest:
                        raise ValueError("恢复数组 SHA 不一致: " + target)
                    destination = dst / target
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and sha(destination) != digest:
                        raise FileExistsError(str(destination))
                    if not destination.exists():
                        shutil.copy2(source, destination)
                    restored.append(
                        dict(
                            destination=target,
                            source=str(source.relative_to(src)),
                            sha256=digest,
                            compared_identity_fields=common,
                        )
                    )
    copied = []
    for i, (rel, digest) in enumerate(sorted(required.items()), 1):
        target = (dst / rel).resolve()
        target.relative_to(dst)
        if not target.exists():
            source = src / rel
            if sha(source) != digest:
                raise ValueError("来源文件 SHA 不一致: " + rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError("目标文件 SHA 不一致: " + rel)
        copied.append(dict(path=rel, sha256=digest, bytes=target.stat().st_size))
        if i % 50 == 0:
            print(f"verified {i}/{len(required)}", flush=True)
    (dst / "IMPORT_RECEIPT.json").write_text(
        json.dumps(
            dict(files=copied, restored_training_arrays=restored), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(status="passed", files=len(copied), restored=len(restored)), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
