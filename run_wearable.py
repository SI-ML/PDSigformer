"""PDSigformer paper experiments: list configurations, train from scratch, or evaluate checkpoints."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["list", "eval", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--profile", help="e.g. shimmer_r4 or pads_timesnet")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a newly trained best.pt; omit to use the historical checkpoint manifest")
    parser.add_argument("--epochs", type=int, help="Override the epoch budget for a short training check, not a full paper rerun")
    cli = parser.parse_args()
    config = json.loads(cli.config.read_text(encoding="utf-8"))
    if cli.mode == "list":
        for key, p in config["profiles"].items():
            print(f"{key:36} {p['dataset']:8} {p['activity']:12} {p['args']['model']}")
        return
    if cli.profile not in config["profiles"]:
        parser.error("--profile 必须是 list 中的配置")
    if cli.output is None:
        parser.error("须显式指定新的 --output 目录")
    if cli.epochs is not None and (cli.mode != "train" or cli.epochs < 1):
        parser.error("--epochs 只能用于训练且必须为正整数")
    if cli.output.exists() and any(
        p.name not in ("stdout.log", "stderr.log", "command.json") for p in cli.output.iterdir()
    ):
        raise FileExistsError("结果目录已有内容，拒绝覆盖: " + str(cli.output))
    cli.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    import torch
    from exp.exp_wearable import Exp_Wearable
    from utils.pds_checkpoint import seed_all, write_json

    profile = config["profiles"][cli.profile]
    settings = dict(
        task_name="classification",
        pred_len=0,
        num_class=2,
        use_gpu=cli.device.startswith("cuda"),
        use_multi_gpu=False,
        gpu=0,
        num_workers=0,
    )
    settings.update(profile["args"])
    settings.update(seed=cli.seed, device=cli.device, swa=profile["swa"])
    settings["eval_batch_size"] = settings.get(
        "eval_batch_size", settings.get("test_batch_size", settings["batch_size"])
    )
    args = SimpleNamespace(**settings)
    torch.set_num_threads(2)
    seed_all(cli.seed)
    torch.use_deterministic_algorithms(profile["kind"] == "pds")
    try:
        experiment = Exp_Wearable(args, profile, cli.artifact_root)
        if cli.mode == "train":
            result = experiment.train(cli.output, cli.epochs)
        else:
            run = next(
                r for r in config["runs"] if r["profile"] == cli.profile and r["seed"] == cli.seed
            )
            result = experiment.test_checkpoint(run, cli.output, cli.checkpoint)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(
            cli.output / "failure.json",
            dict(status="failed", error=str(exc), traceback=traceback.format_exc()),
        )
        raise


if __name__ == "__main__":
    main()
