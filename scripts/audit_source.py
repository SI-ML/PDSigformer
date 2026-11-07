"""核验上游 Python 文件原样保留，并列出新增和修改文件。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "891f65b8a7e77188508fd8c56e10ccdba7fc1e18"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reproduction/SOURCE_AUDIT.json")
    a = p.parse_args()
    manifest_path = ROOT / "reproduction/UPSTREAM_FILES.json"
    has_upstream_commit = (ROOT / ".git").exists() and subprocess.run(
        ["git", "cat-file", "-e", UPSTREAM + "^{commit}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if has_upstream_commit:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", UPSTREAM], cwd=ROOT, text=True
        ).splitlines()
        expected = {}
        for n in names:
            expected[n] = hashlib.sha256(
                subprocess.check_output(["git", "show", UPSTREAM + ":" + n], cwd=ROOT)
            ).hexdigest()
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(dict(commit=UPSTREAM, files=expected), indent=2) + "\n", encoding="utf-8"
            )
    else:
        manifest = json.loads(manifest_path.read_text())
        if manifest["commit"] != UPSTREAM:
            raise ValueError("上游 SHA 清单的提交版本不匹配")
        expected = manifest["files"]
    changed, missing, py_count = [], [], 0
    for name, digest in expected.items():
        path = ROOT / name
        if name.endswith(".py"):
            py_count += 1
        if not path.exists():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    python_changed = [n for n in changed + missing if n.endswith(".py")]
    result = dict(
        upstream_commit=UPSTREAM,
        reference_source="git" if has_upstream_commit else "bundled_manifest",
        upstream_python_files=py_count,
        upstream_python_files_changed=python_changed,
        modified_upstream_files=changed,
        missing_upstream_files=missing,
        passed=not python_changed and not missing,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
