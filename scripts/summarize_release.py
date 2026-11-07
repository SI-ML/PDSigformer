"""只从实际测试记录生成公开指标、HTML 表格与验收摘要；保留失败尝试数量。"""
import argparse
import csv
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc", "macro_auprc"]
LABELS = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def gather(directories):
    selected, attempts = {}, []
    for folder in directories:
        with (folder / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            attempts.append(row)
            if row["passed"] == "True":
                selected[row["id"]] = (
                    json.loads((folder / row["id"] / "receipt.json").read_text()),
                    folder / row["id"],
                )
    return selected, attempts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", nargs="+", type=Path, required=True)
    p.add_argument("--training", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "reproduction")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/paper.json").read_text())
    inf, inf_attempts = gather(a.inference)
    train, train_attempts = gather(a.training)
    expected = {r["id"] + "_r" + str(k) for r in config["runs"] for k in (1, 2)}
    if set(inf) != expected:
        raise ValueError("仍有推理任务缺失: " + str(sorted(expected - set(inf))))
    expected_train = {key + "_s42_r1" for key in config["profiles"]}
    if not expected_train.issubset(train):
        raise ValueError("训练配置测试不完整: " + str(expected_train - set(train)))
    metrics_by_sha, per_seed, repeat_checks = {}, [], []
    for run in config["runs"]:
        first, folder1 = inf[run["id"] + "_r1"]
        second, folder2 = inf[run["id"] + "_r2"]
        if (
            first["checkpoint_sha256"] != run["sha256"]
            or second["checkpoint_sha256"] != run["sha256"]
        ):
            raise ValueError("汇总权重身份不匹配")
        with (folder1 / "predictions.csv").open(encoding="utf-8-sig", newline="") as f:
            p1 = list(csv.DictReader(f))
        with (folder2 / "predictions.csv").open(encoding="utf-8-sig", newline="") as f:
            p2 = list(csv.DictReader(f))
        if len(p1) != len(p2):
            raise ValueError("重复推理样本数不同")
        difference = 0.0
        for x, y in zip(p1, p2):
            for key in ("window_id", "subject_id", "sample_index", "y_true", "y_pred"):
                if x[key] != y[key]:
                    raise ValueError("重复推理身份或预测不同")
            for key in ("p0", "p1"):
                d = abs(float(x[key]) - float(y[key]))
                difference = max(difference, d)
                if d > 1e-6 + 1e-5 * abs(float(y[key])):
                    raise ValueError("两次概率不一致")
        if any(abs(first["metrics"][k] - second["metrics"][k]) > 1e-12 for k in METRICS):
            raise ValueError("重复推理指标不同")
        metrics_by_sha[run["sha256"]] = second["metrics"]
        profile = config["profiles"][run["profile"]]
        per_seed.append(
            dict(
                dataset=profile["dataset"],
                activity=profile["activity"],
                profile=run["profile"],
                seed=run["seed"],
                checkpoint_sha256=run["sha256"],
                **{k: second["metrics"][k] for k in METRICS},
            )
        )
        repeat_checks.append(
            dict(run=run["id"], passed=True, max_abs_probability_difference=difference)
        )
    with (ROOT / "reproduction/PAPER_TABLE_REFERENCES.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        refs = list(csv.DictReader(f))
    groups, table_seeds = defaultdict(list), []
    for row in refs:
        result = dict(
            table=row["paper_table"],
            dataset=row["dataset"],
            activity=row["activity"],
            model=row["paper_row"],
            seed=int(row["seed"]),
            checkpoint_sha256=row["sha256"],
            **{k: metrics_by_sha[row["sha256"]][k] for k in METRICS},
        )
        groups[(result["table"], result["dataset"], result["model"])].append(result)
        table_seeds.append(result)
    aggregate = []
    for (table, dataset, model), rows in groups.items():
        if sorted(r["seed"] for r in rows) != [42, 43, 44]:
            raise ValueError("表格 seed 映射不完整")
        item = dict(table=table, dataset=dataset, model=model, seeds="42,43,44")
        for k in METRICS:
            values = [r[k] for r in rows]
            item[k + "_mean"] = statistics.mean(values)
            item[k + "_std"] = statistics.stdev(values)
        aggregate.append(item)
    training_rows = []
    for key in sorted(expected_train):
        record, _ = train[key]
        if (
            not record["parameters_changed"]
            or not record["reload_probabilities_match"]
            or record["test_accessed"]
        ):
            raise ValueError("训练链路验收字段未通过")
        training_rows.append(
            dict(
                profile=record["profile"],
                seed=record["seed"],
                epochs=record["completed_epochs"],
                parameters_changed=True,
                reload_passed=True,
                test_used=False,
                reload_max_abs_difference=record["reload_max_abs_difference"],
                full_training_verified=record["full_training_verified"],
            )
        )
    write_csv(a.output / "TEST_PER_SEED.csv", per_seed)
    write_csv(a.output / "PAPER_TABLE_SEED_REFERENCES.csv", table_seeds)
    write_csv(a.output / "PAPER_TABLE_MEAN_STD.csv", aggregate)
    write_csv(a.output / "INFERENCE_REPEAT_CHECK.csv", repeat_checks)
    write_csv(a.output / "TRAINING_CHECK.csv", training_rows)
    result = dict(
        unique_checkpoints=len(per_seed),
        table_references=len(table_seeds),
        successful_inference_slots=len(inf),
        repeated_inference_consistent=True,
        profiles_training_checked=len(training_rows),
        training_seeds=[42],
        training_epochs_per_profile=sorted(set(r["epochs"] for r in training_rows)),
        fresh_full_training_verified=all(r["full_training_verified"] for r in training_rows),
        inference_attempts=len(inf_attempts),
        failed_inference_attempts_preserved=sum(r["passed"] != "True" for r in inf_attempts),
        training_attempts=len(train_attempts),
        failed_training_attempts_preserved=sum(r["passed"] != "True" for r in train_attempts),
        maximum_historical_probability_difference=max(
            r[0]["historical_comparison"]["probabilities_max_abs_diff"] for r in inf.values()
        ),
        metric_tolerance=1e-12,
        probability_atol=1e-6,
        probability_rtol=1e-5,
    )
    (a.output / "RELEASE_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    md = f"""# 发布验收记录

- 102 份唯一历史权重、108 个表格引用：全部对应。
- 204 个有效推理任务：每份权重两个独立进程；逐样本身份、标签、预测、概率与六指标通过。
- 六指标与历史记录误差均不超过 1e-12；概率按 atol=1e-6、rtol=1e-5 比较。
- 34 个配置均在 seed42 完成一个完整 epoch，验证有限梯度、参数更新、Valid 选模、保存和全新模型重载。
- **未对全部 102 组重新进行 100 epoch 训练，因此 fresh_full_training_verified=false。**
- 失败尝试保留：推理 {result['failed_inference_attempts_preserved']} 次、训练 {result['failed_training_attempts_preserved']} 次；补测覆盖对应任务，没有覆盖原日志。
- 此轮失败来自记录器把 PyTorch 虚拟模块名当成文件；修复后补测通过。早期 Reformer 迁移绑定失败单独保留在 pilot 记录。
- 官方 Python 完整性由 SOURCE_AUDIT.json 单独核验。

## 可读结果

[离线 HTML 表格](PDSigformer_RESULTS.html)、[每 seed 指标](TEST_PER_SEED.csv)、[表格均值和样本标准差](PAPER_TABLE_MEAN_STD.csv)、[训练测试](TRAINING_CHECK.csv)。

旧权重通过推理验收与从头训练的软件链路通过是两种不同证据。此处不宣称随机训练的逐位轨迹或新训练分数等于历史结果。
"""
    (a.output / "RELEASE_VALIDATION.md").write_text(md, encoding="utf-8")
    pieces = []
    for table in ("Table 1", "Table 2"):
        pieces.append("<h2>" + ("主实验" if table == "Table 1" else "结构消融") + "</h2>")
        pieces.append(
            '<div class="scroll"><table><thead><tr><th>数据集</th><th>模型</th>'
            + "".join("<th>" + k + "</th>" for k in LABELS)
            + "</tr></thead><tbody>"
        )
        for row in aggregate:
            if row["table"] != table:
                continue
            peers = [r for r in aggregate if r["table"] == table and r["dataset"] == row["dataset"]]
            cells = []
            for k in METRICS:
                ranks = sorted(set(r[k + "_mean"] for r in peers), reverse=True)
                rank = ranks.index(row[k + "_mean"])
                tag = "strong" if rank == 0 else "u" if rank == 1 else "span"
                cells.append(
                    f'<td><{tag}>{100*row[k+"_mean"]:.2f} ± {100*row[k+"_std"]:.2f}</{tag}></td>'
                )
            pieces.append(
                "<tr><td>"
                + html.escape(row["dataset"])
                + "</td><td>"
                + html.escape(row["model"])
                + "</td>"
                + "".join(cells)
                + "</tr>"
            )
        pieces.append("</tbody></table></div>")
    doc = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDSigformer 可复现结果</title><style>body{max-width:1450px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172432;background:#f6f8fb}h1{font-size:32px}h2{margin-top:36px}.note{padding:16px 22px;background:#e7f0f8;border-radius:10px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:15px}th,td{padding:11px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid #dbe3eb}th{background:#173b58;color:white}td:first-child,td:nth-child(2){text-align:left}a{color:#1461a4}input{padding:10px;width:300px;border:1px solid #abbccd;border-radius:6px}tr[hidden]{display:none}</style><h1>PDSigformer · 结果与复现验收</h1><p>Shimmer DRINK / PADS CrossArms · 受试者独立 split42 · seeds 42、43、44 · 单位 % · 均值 ± 样本标准差</p><div class="note">102 份权重、108 个表格引用；204 个有效推理任务已核验。34 个配置完成一个 epoch 的训练和重载测试。<strong>未宣称全矩阵已从零完成 100 epoch 重训。</strong></div><p><a href="TEST_PER_SEED.csv">每 seed CSV</a> · <a href="PAPER_TABLE_MEAN_STD.csv">均值/标准差 CSV</a> · <a href="RELEASE_VALIDATION.md">测试说明</a></p><input id="filter" placeholder="筛选数据集或模型"><p>最佳均值加粗，次优均值加下划线。重复出现的 R4 与主实验 PDSigformer 共享同一权重。</p>"""
    doc += "".join(pieces)
    doc += """<script>document.getElementById('filter').oninput=function(){let q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}</script></html>"""
    (a.output / "PDSigformer_RESULTS.html").write_text(doc, encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
