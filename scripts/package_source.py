"""制作不含数据、权重、账号记录和未授权第三方实现的源码 ZIP。"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", "artifacts", "outputs", "checkpoints", "_private"}
EXCLUDED_FILES = {"source_stage.tar.gz", "import_artifacts.py"}
TECH_IMPL = {
    "third_party/tech/model.py",
    "third_party/tech/encoder.py",
    "third_party/tech/augmentation.py",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("拒绝覆盖已有发布包")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(ROOT)
        if (
            set(rel.parts) & EXCLUDED_DIRS
            or rel.as_posix() in EXCLUDED_FILES
            or rel.as_posix() in TECH_IMPL
        ):
            continue
        if path.suffix in (".pt", ".pth", ".npy", ".npz", ".pyc", ".zip", ".gz"):
            continue
        files.append((rel.as_posix(), path))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(a.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel, path in sorted(files):
            data = path.read_bytes()
            archive.writestr("PDSigformer/" + rel, data)
            manifest.append(
                dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        archive.writestr(
            "PDSigformer/SOURCE_PACKAGE_FILES.json", json.dumps(manifest, indent=2) + "\n"
        )
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(".zip.sha256").write_text(
        digest + "  " + a.output.name + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                file=str(a.output),
                files=len(manifest) + 1,
                bytes=a.output.stat().st_size,
                sha256=digest,
            )
        )
    )


if __name__ == "__main__":
    main()
