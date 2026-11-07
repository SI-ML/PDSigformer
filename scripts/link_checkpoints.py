"""按论文表格创建可读的权重链接，R4 与主实验共用同一个 SHA 对象。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    a = p.parse_args()
    with (ROOT / "reproduction/PAPER_TABLE_REFERENCES.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    aliases = []
    for r in rows:
        path = (a.artifact_root / r["target_relative_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        group = "main" if r["paper_table"] == "Table 1" else "ablation"
        folder = a.output / r["dataset"] / group / r["paper_row"]
        folder.mkdir(parents=True, exist_ok=True)
        name = folder / ("seed_" + r["seed"] + path.suffix)
        relative = os.path.relpath(path, name.parent)
        if name.is_symlink():
            if name.resolve() != path:
                raise FileExistsError(str(name))
        elif name.exists():
            raise FileExistsError(str(name))
        else:
            try:
                name.symlink_to(relative)
            except OSError:
                name = name.with_suffix(".json")
                name.write_text(
                    json.dumps(dict(checkpoint=relative, sha256=r["sha256"]), indent=2),
                    encoding="utf-8",
                )
        aliases.append(dict(alias=str(name.relative_to(a.output)), checkpoint_sha256=r["sha256"]))
    (a.output / "INDEX.json").write_text(json.dumps(aliases, indent=2), encoding="utf-8")
    print(
        json.dumps(dict(references=len(rows), unique_checkpoints=len({r["sha256"] for r in rows})))
    )


if __name__ == "__main__":
    main()
