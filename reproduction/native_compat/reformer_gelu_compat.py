"""Reformer 的 CUDA 11.8 GELU 浮点语义兼容层，仅在冻结 Test 中启用。"""
from __future__ import annotations
import ctypes as C
import csv
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReformerGELUCompatibility:
    def __init__(self, root, entry, cfg):
        self.root = Path(root).resolve()
        self.entry, self.cfg = entry, cfg
        self.spec = entry['runtime_compatibility']
        if entry['model'] != 'Reformer' or self.spec['kind'] != 'reformer_gelu_cuda118':
            raise ValueError('GELU 兼容仅适用于完整覆盖的 Reformer 模型')
        if torch.__version__ != '2.8.0+cu128':
            raise ValueError('要求统一 Torch 2.8.0+cu128')
        c = cfg['model_config']['classification_config']
        if c['activation'] != 'gelu':
            raise ValueError('原模型不是 GELU，拒绝改变激活类型')
        self.depth = int(c['e_layers'])
        identities = (self.root / entry['data_root'] / 'prepared' / cfg['task'] / 'test/identities.csv').resolve()
        relative = str(identities.relative_to(self.root))
        if _sha(identities) != entry['data_files'][relative]:
            raise ValueError('冻结 Test 身份文件 SHA 不符')
        with identities.open(encoding='utf-8-sig', newline='') as f:
            samples = sum(1 for _ in csv.DictReader(f))
        if samples <= 0:
            raise ValueError('冻结 Test 样本数为空')
        self.expected_calls = (self.depth + 1) * ((samples + int(cfg['eval_batch_size']) - 1) // int(cfg['eval_batch_size']))
        self.ptx_path = (self.root / self.spec['ptx_path']).resolve()
        self.ptx_path.relative_to(self.root)
        if _sha(self.ptx_path) != self.spec['ptx_sha256']:
            raise ValueError('冻结 PTX 的 SHA 不符')
        self.original = F.gelu
        self.bound = []
        self.active = False
        self.receipt = {'kind': self.spec['kind'], 'torch': torch.__version__,
                        'ptx_path': str(self.ptx_path.relative_to(self.root)),
                        'ptx_sha256': _sha(self.ptx_path),
                        'approximate': 'none', 'formula': 'x * 0.5f * (1.0f + erf(x * float(sqrt(0.5))))',
                        'compiler': 'NVRTC 11.8.89', 'calls': 0, 'elements': 0,
                        'expected_calls': self.expected_calls, 'test_samples': samples,
                        'test_identities_path': relative, 'test_identities_sha256': _sha(identities),
                        'bound_modules': [], 'restored': False,
                        'old_torch_runtime_loaded': False,
                        'historical_outputs_read_or_injected': False,
                        'branch_on_seed_checkpoint_or_predictions': False}

    @staticmethod
    def _state_identity(model):
        # 调用前后校验同一批参数和 buffer 对象、版本及形状，不写模型状态。
        return [(n, id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype))
                for n, t in list(model.named_parameters()) + list(model.named_buffers())]

    def bind_model(self, selected):
        if self.active or self.bound:
            raise RuntimeError('兼容模型重复绑定')
        encoders, classifiers = [], []
        for name, module in selected.named_modules():
            cls = type(module)
            if cls.__module__ == 'layers.Transformer_EncDec' and cls.__name__ == 'EncoderLayer':
                if module.activation is not self.original:
                    raise ValueError('EncoderLayer 原 activation 不是 F.gelu')
                encoders.append((name, module, 'activation'))
            elif cls.__module__ == 'models.Reformer' and cls.__name__ == 'Model':
                if module.task_name != 'classification' or module.act is not self.original:
                    raise ValueError('Reformer 分类输出激活与冻结语义不符')
                classifiers.append((name, module, 'act'))
        if len(encoders) != self.depth or len(classifiers) != 1:
            raise ValueError('GELU 绑定层数或分类器数量与冻结配置不符')
        self.bound = encoders + classifiers
        self.model = selected
        self.state_before = self._state_identity(selected)
        self.receipt['bound_modules'] = [{'module': n or '<model>', 'attribute': a} for n, _, a in self.bound]
        return self

    def _check(self, code, operation):
        if code:
            msg = C.c_char_p()
            self.driver.cuGetErrorName(code, C.byref(msg))
            raise RuntimeError(operation + ': ' + str(code) + ' ' + repr(msg.value))

    def _load(self):
        if torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError('该冻结 PTX 数值见证当前仅验收于 SM89')
        self.driver = C.CDLL('libcuda.so.1')
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
                                              C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                              C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        self.ptx_data = C.create_string_buffer(self.ptx_path.read_bytes())
        self.cuda_module, self.cuda_function = C.c_void_p(), C.c_void_p()
        self._check(self.driver.cuModuleLoadData(C.byref(self.cuda_module), self.ptx_data), 'cuModuleLoadData')
        self._check(self.driver.cuModuleGetFunction(C.byref(self.cuda_function), self.cuda_module, b'gelu_exact'), 'cuModuleGetFunction')
        v = C.c_int()
        self._check(self.driver.cuDriverGetVersion(C.byref(v)), 'cuDriverGetVersion')
        self.receipt.update(driver_api_version=v.value, gpu_name=torch.cuda.get_device_name(), compute_capability=[8, 9])

    def gelu(self, input, approximate='none'):
        if not self.active or approximate != 'none':
            raise RuntimeError('只允许启用上下文内原 approximate=none 语义')
        if torch.is_grad_enabled() or not input.is_cuda or input.dtype != torch.float32:
            raise RuntimeError('该兼容算子只允许 CUDA float32 的 no_grad Test')
        # 通用逐元素计算，输入来自当前模型。连续化不改变数值与随机流。
        x = input.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        if n:
            xp, yp, count = C.c_void_p(x.data_ptr()), C.c_void_p(y.data_ptr()), C.c_longlong(n)
            arguments = (C.c_void_p * 3)(C.cast(C.byref(xp), C.c_void_p), C.cast(C.byref(yp), C.c_void_p), C.cast(C.byref(count), C.c_void_p))
            self._check(self.driver.cuLaunchKernel(self.cuda_function, (n + 255) // 256, 1, 1,
                        256, 1, 1, 0, C.c_void_p(torch.cuda.current_stream(input.device).cuda_stream), arguments, None), 'cuLaunchKernel')
        self.receipt['calls'] += 1
        self.receipt['elements'] += n
        return y

    def __enter__(self):
        if self.active or not self.bound or F.gelu is not self.original:
            raise RuntimeError('GELU 上下文状态不合法')
        self._load()
        try:
            self.active = True
            F.gelu = self.gelu
            for _, module, attribute in self.bound:
                setattr(module, attribute, self.gelu)
            return self
        except Exception:
            self._restore()
            raise

    def _restore(self):
        for _, module, attribute in self.bound:
            setattr(module, attribute, self.original)
        F.gelu = self.original
        self.active = False
        self.receipt['restored'] = all(getattr(m, a) is self.original for _, m, a in self.bound) and F.gelu is self.original
        self.receipt['parameter_buffer_objects_versions_unchanged'] = self._state_identity(self.model) == self.state_before

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            torch.cuda.synchronize()
        finally:
            self._restore()
        if not self.receipt['restored'] or not self.receipt['parameter_buffer_objects_versions_unchanged']:
            raise RuntimeError('GELU 上下文退出后模型状态或原函数恢复失败')
        self.receipt['calls_match_expected'] = self.receipt['calls'] == self.expected_calls
        if exc_type is None and not self.receipt['calls_match_expected']:
            raise RuntimeError('GELU 实际调用次数与原模型层数、完整 Test 批次数不符')
        return False
