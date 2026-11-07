"""固定 Autoformer Test 的 cuFFT10.9 兼容层，保持原 ATen 末轴 FFT 语义。"""
import ctypes
import hashlib
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


class AutoformerCuFFTCompatibility:
    def __init__(self, root, entry, config):
        import torch
        self.torch = torch
        self.root = Path(root).resolve()
        spec = entry['runtime_compatibility']
        if entry['model'] != 'Autoformer' or spec['kind'] != 'autoformer_lastdim_rfft_irfft_cufft109':
            raise ValueError('兼容层仅适用于已冻结 Autoformer 条目')
        if torch.__version__ != '2.8.0+cu128':
            raise ValueError('兼容层要求已锁定 Torch2.8.0+cu128')
        constructor = config['model_config']['constructor_config']
        actual = {key:int(constructor[key]) for key in ('seq_len','d_model','n_heads','e_layers')}
        actual.update(eval_batch_size=int(config['eval_batch_size']), test_samples=int(config['split_shapes']['test'][0]))
        if actual != spec['scope'] or actual['d_model'] % actual['n_heads']:
            raise ValueError('兼容层实参范围与冻结构造或评测配置不同')
        for relative, expected in entry['runtime_files'].items():
            path = (self.root/relative).resolve()
            if not path.is_relative_to(self.root) or file_sha(path) != expected:
                raise ValueError('兼容层文件越界或 SHA 不一致: '+relative)
        self.path = (self.root/spec['library_path']).resolve()
        if not self.path.is_relative_to(self.root):
            raise ValueError('数值库路径越出集中根')
        if file_sha(self.path) != 'ca5c6f81d584906b4d32b984ad8704dd65bf75bdab4334ed22ce7eef7501a95a':
            raise ValueError('数值库 SHA 与已验证官方动态库不同')
        self.length = actual['seq_len']; self.heads = actual['n_heads']
        self.head_width = actual['d_model']//self.heads; self.layers = actual['e_layers']
        batch = actual['eval_batch_size']; count = actual['test_samples']
        self.batches = [min(batch,count-i) for i in range(0,count,batch)]
        self.lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        I, LL, SZ, P = ctypes.c_int, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_void_p
        signatures = {
            'cufftGetVersion': [ctypes.POINTER(I)], 'cufftCreate': [ctypes.POINTER(I)],
            'cufftSetAutoAllocation': [I, I],
            'cufftXtMakePlanMany': [I,I,ctypes.POINTER(LL),ctypes.POINTER(LL),LL,LL,I,ctypes.POINTER(LL),LL,LL,I,LL,ctypes.POINTER(SZ),I],
            'cufftSetStream': [I,P], 'cufftSetWorkArea': [I,P],
            'cufftXtExec': [I,P,P,I], 'cufftDestroy': [I],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = I
        version = I(); self.check(self.lib.cufftGetVersion(ctypes.byref(version)), 'version')
        if version.value != spec['cufft_version'] or version.value != 10900:
            raise ValueError('实际 cuFFT 版本不同')
        self.calls = []
        self.receipt = {
            'enabled': True, 'kind': spec['kind'], 'library_path': str(self.path),
            'library_sha256': file_sha(self.path), 'cufft_version': version.value,
            'distribution_version': '10.9.0.58', 'scope': actual,
            'module_path': str(Path(__file__).resolve()), 'module_sha256': file_sha(__file__),
            'runtime_files_sha256': entry['runtime_files'], 'calls': self.calls,
            'original_static_and_dynamic_library_bytes_identical_claim': False,
            'selection_uses_seed_or_predictions': False, 'frozen_source_changed': False,
            'mathematical_transform_unchanged': True, 'observer_hooks_used': False,
            'semantics_reference': 'https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/cuda/SpectralOps.cpp',
        }

    @staticmethod
    def check(code, op):
        if code != 0:
            raise RuntimeError(op+' cuFFT status '+str(code))

    def __enter__(self):
        self.old_rfft, self.old_irfft = self.torch.fft.rfft, self.torch.fft.irfft
        self.torch.fft.rfft, self.torch.fft.irfft = self.rfft, self.irfft
        return self

    def __exit__(self, error_type, error, traceback):
        self.torch.fft.rfft, self.torch.fft.irfft = self.old_rfft, self.old_irfft
        self.receipt['functions_restored'] = True
        self.receipt['call_count'] = len(self.calls)
        if self.batches is not None:
            expected = len(self.batches)*self.layers*3
            self.receipt['expected_call_count'] = expected
            self.receipt['expected_call_count_matched'] = len(self.calls) == expected
            if error_type is None:
                assert len(self.calls) == expected
        return False

    def transform(self, x, inverse, n, dim, norm, out):
        torch = self.torch
        assert x.is_cuda and not x.requires_grad and x.ndim == 4 and x.is_contiguous()
        assert n is None and dim in (-1,3) and norm in (None,'backward') and out is None
        assert x.dtype == (torch.complex64 if inverse else torch.float32)
        batch, heads, width, last = x.shape
        assert (heads,width) == (self.heads,self.head_width)
        assert last == (self.length//2+1 if inverse else self.length)
        if self.batches is not None:
            call = len(self.calls); group = call//3
            assert group < len(self.batches)*self.layers
            assert batch == self.batches[group//self.layers]
            assert inverse == (call%3 == 2)
        # 原 ATen C2R 必须 clone，避免 cuFFT 覆写调用者输入。
        source = x.clone(memory_format=torch.contiguous_format) if inverse else x
        source = source.reshape(batch*heads*width, last)
        result = torch.empty((batch*heads*width, self.length if inverse else self.length//2+1),
                             dtype=torch.float32 if inverse else torch.complex64, device=x.device)
        handle = ctypes.c_int(); self.check(self.lib.cufftCreate(ctypes.byref(handle)), 'create')
        stream = torch.cuda.current_stream(x.device)
        try:
            self.check(self.lib.cufftSetAutoAllocation(handle,0), 'autoalloc')
            size = (ctypes.c_longlong*1)(self.length); workspace_size = ctypes.c_size_t()
            self.check(self.lib.cufftXtMakePlanMany(handle,1,size,None,1,1,4 if inverse else 0,
                       None,1,1,0 if inverse else 4,batch*heads*width,ctypes.byref(workspace_size),4), 'plan')
            workspace = torch.empty((workspace_size.value,), dtype=torch.uint8, device=x.device)
            self.check(self.lib.cufftSetStream(handle,ctypes.c_void_p(stream.cuda_stream)), 'stream')
            self.check(self.lib.cufftSetWorkArea(handle,ctypes.c_void_p(workspace.data_ptr())), 'workspace')
            self.check(self.lib.cufftXtExec(handle,ctypes.c_void_p(source.data_ptr()),ctypes.c_void_p(result.data_ptr()),
                                          1 if inverse else -1), 'execute')
            # 原 ATen _fft_apply_normalization 是 output.mul_(1.0/N)。
            if inverse:
                result.mul_(1.0/self.length)
            stream.synchronize()
            self.calls.append({'operation': 'irfft' if inverse else 'rfft', 'input_shape': list(x.shape),
                               'input_stride': list(x.stride()), 'plan_batch': batch*heads*width,
                               'plan_input_stride': list(source.stride()), 'workspace_bytes': workspace_size.value,
                               'direction': 1 if inverse else -1, 'normalization': 1.0/self.length if inverse else 1.0,
                               'input_cloned': inverse, 'null_embeds': True, 'current_stream': stream.cuda_stream})
            return result.reshape(batch,heads,width,result.shape[-1])
        finally:
            self.check(self.lib.cufftDestroy(handle), 'destroy')

    def rfft(self, x, n=None, dim=-1, norm=None, *, out=None):
        return self.transform(x,False,n,dim,norm,out)

    def irfft(self, x, n=None, dim=-1, norm=None, *, out=None):
        return self.transform(x,True,n,dim,norm,out)
