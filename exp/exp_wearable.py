"""在官方分类实验类上增加固定可穿戴协议，官方入口与实现保持不变。"""
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
from exp.exp_classification import Exp_Classification
from data_provider.wearable_loader import WearableLoader, make_loader
from utils.pds_checkpoint import (
    load_trusted,
    metrics,
    save_predictions,
    sha256,
    state_hash,
    write_json,
)
from utils.pds_runtime import (
    build_model,
    evaluation_rng,
    evaluation_seed,
    historical_context,
    non_tensor_state,
    restore_non_tensor,
)


class Exp_Wearable(Exp_Classification):
    def __init__(self, args, profile, artifact_root):
        self.profile = profile
        self.artifact_root = Path(artifact_root).resolve()
        self.datasets = {}
        super().__init__(args)

    def _acquire_device(self):
        device = torch.device(self.args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但 CUDA 不可用；不自动改变运行设备")
        return device

    def _build_model(self):
        return build_model(self.profile, self.args)

    def _get_data(self, flag):
        split = {"val": "valid", "vali": "valid"}.get(flag.lower(), flag.lower())
        if split not in self.datasets:
            self.datasets[split] = WearableLoader(
                self.artifact_root / self.profile["data_relative"],
                args=self.args,
                flag=split,
                expected_files=self.profile.get("data_hashes"),
            )
        data = self.datasets[split]
        if (data.max_seq_len, data.num_channels) != (self.args.seq_len, self.args.enc_in):
            raise ValueError("配置与实际数据形状不同")
        batch = self.args.batch_size if split == "train" else self.args.eval_batch_size
        es = evaluation_seed(self.profile, self.args.seed, split)
        return data, make_loader(
            data,
            batch,
            self.args.seed if split == "train" or es is None else es,
            train=split == "train",
        )

    def _select_optimizer(self):
        cls = torch.optim.RAdam if self.profile["optimizer"] == "RAdam" else torch.optim.Adam
        return cls(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.0)

    def selected_model(self):
        return self.swa_model if self.swa else self.model

    def evaluate_split(self, split, model=None):
        data, loader = self._get_data(split)
        model = self.selected_model() if model is None else model
        was_training = model.training
        model.eval()
        losses, probabilities, outputs = [], [], []
        with evaluation_rng(evaluation_seed(self.profile, self.args.seed, split)), torch.no_grad():
            for x, y, mask in loader:
                logits = model(x.float().to(self.device), mask.float().to(self.device), None, None)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("非有限 logits")
                losses.append(float(self._select_criterion()(logits.cpu(), y.long())))
                if self.profile["kind"] == "pds":
                    outputs.append(logits)
                else:
                    probabilities.append(logits.softmax(dim=1).cpu().numpy())
            if outputs:
                prob = torch.cat(outputs).softmax(dim=1).cpu().numpy()
            else:
                prob = np.concatenate(probabilities)
        model.train(was_training)
        return dict(**metrics(data.y, prob), loss=float(np.mean(losses))), prob, data

    def environment(self):
        root = Path(__file__).resolve().parents[1]
        imported = {}
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if root in path.parents and path.suffix == ".py" and path.is_file():
                    imported[path.relative_to(root).as_posix()] = sha256(path)
        return dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numpy=np.__version__,
            device=str(self.device),
            device_name=torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else platform.processor(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32=False,
            imported_source_sha256=imported,
        )

    def test_checkpoint(self, run, output, checkpoint=None):
        output = Path(output)
        started = time.perf_counter()
        historical = checkpoint is None
        cp = self.artifact_root / run["checkpoint"] if historical else Path(checkpoint)
        saved = load_trusted(cp, run["sha256"] if historical else None)
        selected = self.selected_model()
        initial = state_hash(self.model)
        if historical and self.profile["kind"] == "pds":
            # 原位置编码是非训练 buffer；整体初始化摘要验证其与原实验一致。
            with torch.no_grad():
                for name, buf in self.model.named_buffers():
                    if name.endswith(".pe") and name in saved["model"]:
                        buf.copy_(saved["model"][name])
            restored_initial = state_hash(self.model)
            if restored_initial != run["initial_state_sha256"]:
                raise ValueError("PDS 初始化与历史见证不一致")
            write_json(
                output / "initialization.json",
                dict(
                    before=initial,
                    restored_static_buffers=restored_initial,
                    expected=run["initial_state_sha256"],
                    passed=True,
                ),
            )
        if historical:
            state = saved["model"] if self.profile["kind"] == "pds" else saved["state_dict"]
        else:
            if (
                saved.get("format") != "pdsigformer_training_v1"
                or saved["profile"] != self.profile["id"]
                or saved["seed"] != self.args.seed
            ):
                raise ValueError("训练权重的配置或 seed 不匹配")
            state = saved["state_dict"]
        restore_non_tensor(selected, saved.get("construction_metadata", non_tensor_state(selected)))
        selected.load_state_dict(state, strict=True)
        if historical and run.get("parameter_count") is not None:
            if sum(p.numel() for p in self.model.parameters()) != run["parameter_count"]:
                raise ValueError("参数数量不匹配")
        context = (
            historical_context(self.artifact_root, run, selected, self.device)
            if historical
            else __import__("contextlib").nullcontext()
        )
        with context:
            result, prob, data = self.evaluate_split("test", selected)
        if hasattr(context, "receipt"):
            write_json(output / "runtime_compatibility.json", context.receipt)
        save_predictions(output / "predictions.csv", data, prob)
        audit = {}
        if historical:
            ref = self.artifact_root / run["reference_predictions"]
            if sha256(ref) != run["reference_predictions_sha256"]:
                raise ValueError("原预测证据 SHA 不符")
            with ref.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            rp = np.array([[float(r["p0"]), float(r["p1"])] for r in rows])
            identities = [json.loads(r["identity_json"]) for r in rows]
            same_identity = len(rows) == len(data) and all(
                str(v["window_id"]) == str(data.window_ids[i])
                and str(v["subject_id"]) == str(data.subject_ids[i])
                and int(rows[i]["y_true"]) == int(data.y[i])
                and int(rows[i]["sample_index"]) == i
                for i, v in enumerate(identities)
            )
            differences = {k: abs(result[k] - v) for k, v in run["expected_metrics"].items()}
            audit = dict(
                identity_equal=same_identity,
                probabilities_within_tolerance=bool(np.allclose(prob, rp, atol=1e-6, rtol=1e-5)),
                probabilities_max_abs_diff=float(np.max(np.abs(prob - rp))),
                predictions_equal=bool(np.array_equal(prob.argmax(1), rp.argmax(1))),
                metric_abs_differences=differences,
            )
            audit["passed"] = (
                same_identity
                and audit["probabilities_within_tolerance"]
                and audit["predictions_equal"]
                and all(v <= 1e-12 for v in differences.values())
            )
        receipt = dict(
            status="completed",
            passed=audit.get("passed", True),
            run=run["id"],
            profile=self.profile["id"],
            seed=self.args.seed,
            checkpoint_sha256=sha256(cp),
            checkpoint_best_epoch=saved.get("best_epoch", saved.get("epoch")),
            loaded_state_sha256=state_hash(selected),
            metrics=result,
            historical_comparison=audit,
            environment=self.environment(),
            elapsed_seconds=time.perf_counter() - started,
            training_performed=False,
            validation_read=False,
            reference_probabilities_used_as_input=False,
        )
        write_json(output / "metrics.json", result)
        write_json(output / "receipt.json", receipt)
        if not receipt["passed"]:
            raise RuntimeError("历史推理对照未通过，详见 receipt.json")
        return receipt

    def train(self, setting, epochs=None):
        output = Path(setting)
        started = time.perf_counter()
        train, loader = self._get_data("train")
        valid, _ = self._get_data("valid")
        if set(train.subject_ids) & set(valid.subject_ids):
            raise ValueError("Train/Valid 存在受试者重叠")
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        budget = self.profile["epochs"] if epochs is None else epochs
        patience = self.profile["patience"]
        initial = state_hash(self.model)
        branch_initial = {
            name: state_hash(getattr(self.model, name))
            for name in ("frontend", "projection", "backbone")
            if hasattr(self.model, name)
            and any(True for _ in getattr(self.model, name).parameters())
        }
        branch_gradients = {}
        best, best_epoch, stale = -float("inf"), 0, 0
        rows = []
        write_json(
            output / "effective_config.json",
            dict(
                profile=self.profile,
                args=vars(self.args),
                epochs=budget,
                patience=patience,
                optimizer_defaults=optimizer.defaults,
                environment=self.environment(),
                initial_state_sha256=initial,
                test_used_for_selection=False,
                checkpoint_selection="Valid Macro-F1",
                delta=1e-5,
                full_training=budget == self.profile["epochs"],
            ),
        )
        for epoch in range(1, budget + 1):
            tick = time.perf_counter()
            self.model.train()
            losses, grad_norms = [], []
            for x, y, mask in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(
                    x.float().to(self.device), mask.float().to(self.device), None, None
                )
                loss = criterion(logits, y.long().to(self.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("非有限训练 loss")
                loss.backward()
                if epoch == 1 and not losses:
                    for name in branch_initial:
                        value = sum(
                            float(p.grad.detach().abs().sum())
                            for p in getattr(self.model, name).parameters()
                            if p.grad is not None
                        )
                        if not np.isfinite(value) or value <= 0:
                            raise RuntimeError("分支未获得有限非零梯度: " + name)
                        branch_gradients[name] = value
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("非有限梯度")
                grad_norms.append(float(grad_norm))
                optimizer.step()
                losses.append(float(loss))
            if self.swa:
                self.swa_model.update_parameters(self.model)
            selected = self.selected_model()
            score, _, _ = self.evaluate_split("valid", selected)
            if score["macro_f1"] >= best + 1e-5:
                best, best_epoch, stale = score["macro_f1"], epoch, 0
                saved = dict(
                    format="pdsigformer_training_v1",
                    profile=self.profile["id"],
                    seed=self.args.seed,
                    epoch=epoch,
                    valid_macro_f1=best,
                    state_dict={
                        k: v.detach().cpu().clone() for k, v in selected.state_dict().items()
                    },
                    construction_metadata=non_tensor_state(selected),
                    optimizer=optimizer.state_dict(),
                    args=vars(self.args),
                    environment=self.environment(),
                )
                torch.save(saved, output / "best.pt")
            else:
                stale += 1
            rows.append(
                dict(
                    epoch=epoch,
                    train_loss=float(np.mean(losses)),
                    valid_macro_f1=score["macro_f1"],
                    best_epoch=best_epoch,
                    gradient_norm_max=max(grad_norms),
                    seconds=time.perf_counter() - tick,
                )
            )
            with (output / "epochs.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(rows[-1]), flush=True)
            if stale >= patience:
                break
        # 严格读取新保存权重，并与另一全新模型的重载结果逐项对照。
        saved = load_trusted(output / "best.pt")
        selected = self.selected_model()
        restore_non_tensor(selected, saved["construction_metadata"])
        selected.load_state_dict(saved["state_dict"], strict=True)
        score, before, data = self.evaluate_split("valid", selected)
        fresh = build_model(self.profile, self.args).to(self.device)
        if self.swa:
            fresh = torch.optim.swa_utils.AveragedModel(fresh)
        restore_non_tensor(fresh, saved["construction_metadata"])
        fresh.load_state_dict(saved["state_dict"], strict=True)
        after_score, after, _ = self.evaluate_split("valid", fresh)
        same = bool(np.allclose(before, after, atol=1e-6, rtol=1e-5))
        changed = initial != state_hash(self.model)
        if not same or not changed or max(r["gradient_norm_max"] for r in rows) <= 0:
            raise RuntimeError("训练更新或 checkpoint 重载验证失败")
        save_predictions(output / "valid_predictions.csv", data, after)
        receipt = dict(
            status="completed",
            passed=True,
            profile=self.profile["id"],
            seed=self.args.seed,
            completed_epochs=len(rows),
            configured_epochs=budget,
            original_epochs=self.profile["epochs"],
            full_training_verified=len(rows) == self.profile["epochs"],
            best_epoch=best_epoch,
            best_valid_macro_f1=best,
            reloaded_valid_metrics=after_score,
            checkpoint_sha256=sha256(output / "best.pt"),
            train_samples=len(train),
            valid_samples=len(valid),
            test_accessed=False,
            parameters_changed=changed,
            reload_probabilities_match=same,
            first_step_branch_gradient_l1=branch_gradients,
            branch_parameters_changed={
                name: state_hash(getattr(self.model, name)) != digest
                for name, digest in branch_initial.items()
            },
            reload_max_abs_difference=float(np.max(np.abs(before - after))),
            elapsed_seconds=time.perf_counter() - started,
            environment=self.environment(),
        )
        write_json(output / "receipt.json", receipt)
        return receipt
