"""固定论文配置的模型构造、非张量状态及历史推理运行库。"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils.pds_checkpoint import seed_all, sha256


def build_model(profile, args):
    if profile["kind"] == "pds":
        from models.PDSigformer import Model

        config = deepcopy(args)
        config.pds_variant = "residual"
        if profile["role"] == "R3":
            config.no_inter_attn = False
        model = Model(config).float()
        if profile["role"] == "R1":
            model = model.backbone
        elif profile["role"] == "R3":
            for layer in model.backbone.encoder.attn_layers:
                layer.attention.inter_attention = None
        return model
    name = args.model
    package = "models" if name in ("TimesNet", "TeCh") else "third_party.medformer_paper.models"
    module = importlib.import_module(package + "." + name)
    return module.Model(args, **profile.get("constructor_kwargs", {})).float()


def non_tensor_state(model):
    while isinstance(model, torch.optim.swa_utils.AveragedModel):
        model = model.module
    modules = {}
    for name, obj in model.named_modules():
        if type(obj).__name__ in ("FourierBlock", "FourierCrossAttention"):
            modules[name] = {
                k: [int(v) for v in getattr(obj, k)]
                for k in ("index", "index_q", "index_kv")
                if hasattr(obj, k)
            }
    return dict(format_version=1, modules=modules)


def restore_non_tensor(model, state):
    while isinstance(model, torch.optim.swa_utils.AveragedModel):
        model = model.module
    modules = dict(model.named_modules())
    for name, fields in state.get("modules", {}).items():
        obj = modules[name]
        for key, values in fields.items():
            if key not in ("index", "index_q", "index_kv") or not hasattr(obj, key):
                raise ValueError("未知非张量状态")
            setattr(obj, key, list(values))
    if non_tensor_state(model) != state:
        raise ValueError("非张量状态不匹配")


def evaluation_seed(profile, seed, split):
    if profile["kind"] == "pds":
        return None
    value = f"baseline-eval-v1|{seed}|{profile['args']['model']}|{profile['task']}|{split}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


@contextmanager
def evaluation_rng(seed):
    if seed is None:
        yield
        return
    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def historical_context(artifact_root, run, selected, device):
    """兼容库只用于锁定的历史 Test；训练不使用无梯度数值兼容算子。"""
    if run["historical_manifest"] != "baseline78_unified_v4":
        return nullcontext()
    root = Path(artifact_root)
    entries = json.loads((root / "manifests/baseline78_unified_v4.json").read_text())["runs"]
    row = next(r for r in entries if r["sha"] == run["sha256"])
    spec = row.get("runtime_compatibility")
    if not spec:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("该历史精确推理需要 CUDA 和记录中的数值兼容库")
    for rel, expected in row["runtime_files"].items():
        if sha256(root / rel) != expected:
            raise ValueError("历史运行库 SHA 不符: " + rel)
    module_path = (
        Path(__file__).resolve().parents[1]
        / "reproduction/native_compat"
        / Path(spec["module_path"]).name
    )
    if sha256(module_path) != row["runtime_files"][spec["module_path"]]:
        raise ValueError("发布版历史兼容源码与冻结来源不一致")
    module_spec = importlib.util.spec_from_file_location("pds_native_compat", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    names = dict(
        TimesNet="TimesNetCuFFTCompatibility",
        Autoformer="AutoformerCuFFTCompatibility",
        Reformer="ReformerGELUCompatibility",
    )
    config = json.loads((root / row["test_config"]).read_text())
    result = getattr(module, names[row["model"]])(root, row, config)
    if row["model"] == "Reformer":
        # 冻结兼容器使用旧模块绝对名称；迁移后按同一类和同一激活做严格绑定。
        from third_party.medformer_paper.layers.Transformer_EncDec import EncoderLayer
        from third_party.medformer_paper.models.Reformer import Model

        encoders, classifiers = [], []
        for name, obj in selected.named_modules():
            if type(obj) is EncoderLayer:
                if obj.activation is not result.original:
                    raise ValueError("Reformer 编码层激活发生改变")
                encoders.append((name, obj, "activation"))
            elif type(obj) is Model:
                if obj.task_name != "classification" or obj.act is not result.original:
                    raise ValueError("Reformer 分类头激活发生改变")
                classifiers.append((name, obj, "act"))
        if len(encoders) != result.depth or len(classifiers) != 1:
            raise ValueError("Reformer 兼容绑定数量不符")
        result.bound = encoders + classifiers
        result.model = selected
        result.state_before = result._state_identity(selected)
        result.receipt["bound_modules"] = [
            {"module": n or "<model>", "attribute": k} for n, _, k in result.bound
        ]
        result.receipt["module_namespace"] = "third_party.medformer_paper"
    return result
