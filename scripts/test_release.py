"""Run inference or training checks in independent processes and record live task status in CSV."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=["inference", "training"])
    p.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", default="0", help="每张卡一个子进程，如 0,1；可传 CPU")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1, help="训练测试默认一个完整 epoch；完整重训用 100")
    p.add_argument("--profiles", nargs="*")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--minimum-free-mib", type=int, default=10000)
    p.add_argument("--retry-failed-from", type=Path, help="仅重试已完成测试目录中的失败任务，原记录保留")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("测试目录已存在，须使用新的目录")
    a.output.mkdir(parents=True)
    config = json.loads((ROOT / "configs/paper.json").read_text())
    selected = set(a.profiles or config["profiles"])
    unknown = selected - set(config["profiles"])
    if unknown:
        raise ValueError("未知 profile: " + repr(unknown))
    seeds = a.seeds or ([42, 43, 44] if a.suite == "inference" else [42])
    jobs = [
        dict(profile=r["profile"], seed=r["seed"], repeat=k, id=r["id"] + "_r" + str(k))
        for k in range(1, a.repeats + 1 if a.suite == "inference" else 2)
        for r in config["runs"]
        if r["profile"] in selected and r["seed"] in seeds
    ]
    if a.retry_failed_from:
        summary = a.retry_failed_from / "summary.csv"
        if not summary.is_file():
            raise ValueError("来源队列尚未完整结束，没有 summary.csv")
        with summary.open(newline="", encoding="utf-8") as f:
            failed = [r for r in csv.DictReader(f) if r["passed"] != "True"]
        jobs = [
            dict(profile=r["profile"], seed=int(r["seed"]), repeat=int(r["repeat"]), id=r["id"])
            for r in failed
        ]
        if not jobs:
            raise ValueError("没有失败任务需要重试")
    write_json = lambda p, v: p.write_text(
        json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_json(
        a.output / "plan.json",
        dict(
            suite=a.suite,
            epochs=a.epochs,
            jobs=jobs,
            gpus=a.gpus,
            inference_uses_reference_as_input=False,
            training_tests_use_test_data=False,
        ),
    )
    gpu_list = a.gpus.split(",")

    def worker(gpu, chunk):
        rows = []
        for job in chunk:
            folder = a.output / job["id"]
            folder.mkdir()
            command = [
                sys.executable,
                "-B",
                str(ROOT / "run_wearable.py"),
                "eval" if a.suite == "inference" else "train",
                "--profile",
                job["profile"],
                "--seed",
                str(job["seed"]),
                "--artifact-root",
                str(a.artifact_root),
                "--output",
                str(folder),
                "--device",
                "cpu" if gpu.upper() == "CPU" else "cuda:0",
            ]
            if a.suite == "training":
                command += ["--epochs", str(a.epochs)]
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=gpu,
                PYTHONHASHSEED=str(job["seed"]),
                PYTHONDONTWRITEBYTECODE="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="2",
            )
            gpu_record = None
            if gpu.upper() != "CPU":
                for attempt in range(120):
                    output = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            gpu,
                            "--query-gpu=uuid,name,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip()
                    gpu_record = output
                    free = int(output.split(",")[2].strip())
                    if free >= a.minimum_free_mib:
                        break
                    print(f"waiting GPU {gpu}: free {free} MiB", flush=True)
                    time.sleep(30)
                else:
                    raise RuntimeError("GPU 显存不足，保持失败记录，不改变 batch")
            write_json(
                folder / "command.json", dict(command=command, gpu=gpu, gpu_snapshot=gpu_record)
            )
            tick = time.perf_counter()
            with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open(
                "w"
            ) as stderr:
                code = subprocess.call(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            receipt = (
                json.loads((folder / "receipt.json").read_text())
                if (folder / "receipt.json").exists()
                else {}
            )
            row = dict(
                **job,
                gpu=gpu,
                exit_code=code,
                passed=code == 0 and receipt.get("passed") is True,
                seconds=round(time.perf_counter() - tick, 3),
                metrics=json.dumps(
                    receipt.get("metrics", receipt.get("reloaded_valid_metrics", {}))
                ),
                checkpoint_sha256=receipt.get("checkpoint_sha256", ""),
            )
            rows.append(row)
            with (a.output / ("gpu_" + gpu + ".csv")).open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as pool:
        pending = [
            pool.submit(worker, gpu, jobs[i :: len(gpu_list)]) for i, gpu in enumerate(gpu_list)
        ]
        for future in as_completed(pending):
            all_rows.extend(future.result())
    all_rows.sort(key=lambda r: r["id"])
    with (a.output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    result = dict(
        suite=a.suite,
        total=len(all_rows),
        passed=sum(r["passed"] for r in all_rows),
        failed=sum(not r["passed"] for r in all_rows),
        completed=True,
    )
    write_json(a.output / "summary.json", result)
    print(json.dumps(result), flush=True)
    if result["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
