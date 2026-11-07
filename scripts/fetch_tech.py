"""从固定官方提交获取 TeCh，仅调整三个模块的相对导入并校验 SHA。"""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "third_party/VENDOR_MANIFEST.json").read_text())
    for row in manifest["files"]:
        if not row["vendored_path"].startswith("tech/"):
            continue
        target = ROOT / "third_party" / row["vendored_path"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["vendored_sha256"]:
                raise ValueError("已有 TeCh 文件与固定版本不一致，拒绝覆盖: " + str(target))
            print("verified", target.name)
            continue
        for attempt in range(3):
            try:
                raw = urllib.request.urlopen(row["source_url"], timeout=30).read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
        if hashlib.sha256(raw).hexdigest() != row["original_sha256"]:
            raise ValueError("TeCh 上游文件 SHA 不一致")
        text = raw.decode("utf-8")
        for change in row["patches"]:
            text = text.replace(change["before"], change["after"])
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() != row["vendored_sha256"]:
            raise ValueError("TeCh 导入适配后的 SHA 不一致")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print("downloaded and verified", target.name)
    print("TeCh 已安装；来源和许可证状态见 third_party/VENDOR_MANIFEST.json。")


if __name__ == "__main__":
    main()
