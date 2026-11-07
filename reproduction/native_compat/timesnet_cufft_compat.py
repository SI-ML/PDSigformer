"""固定 TimesNet Test 的 cuFFT10.9 数值兼容层；不修改模型源码或超参数。"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class TimesNetCuFFTCompatibility:
    """只包装已冻结 Test 中 TimesNet 原 FP32、一维、不归一化的 rfft 调用。"""

    def __init__(self, root, entry, config):
        import torch
        self.torch = torch
        self.root = Path(root).resolve()
        self.entry = entry
        spec = entry['runtime_compatibility']
        if entry['model'] != 'TimesNet' or spec['kind'] != 'timesnet_cufft109_rfft':
            raise ValueError('兼容层仅适用于已冻结 TimesNet 条目')
        constructor = config['model_config']['constructor_config']
        actual = {'seq_len':int(constructor['seq_len']), 'd_model':int(constructor['d_model']),
                  'e_layers':int(constructor['e_layers']), 'eval_batch_size':int(config['eval_batch_size']),
                  'test_samples':int(config['split_shapes']['test'][0])}
        if actual != spec['scope']:
            raise ValueError('兼容层实参范围与冻结原配置不一致')
        if torch.__version__ != '2.8.0+cu128':
            raise ValueError('本兼容层仅在已验证统一 Torch2.8.0+cu128 中使用')
        for rel, expected in entry['runtime_files'].items():
            path = (self.root/rel).resolve()
            if not path.is_relative_to(self.root) or file_sha(path) != expected:
                raise ValueError('兼容层文件越界或 SHA 不一致: '+rel)
        self.path = (self.root/spec['library_path']).resolve()
        self.lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        self._declare()
        version = ctypes.c_int()
        self._check(self.lib.cufftGetVersion(ctypes.byref(version)), 'cufftGetVersion')
        if version.value != spec['cufft_version'] or version.value != 10900:
            raise ValueError('实际加载 cuFFT 版本与冻结版本不一致')
        self.scope = actual
        batch = actual['eval_batch_size']; samples = actual['test_samples']
        self.expected_batches = [min(batch, samples-start) for start in range(0, samples, batch)]
        self.calls = []
        self.old_rfft = None
        self.receipt = {'enabled':True,'kind':spec['kind'],'module_path':str(Path(__file__).resolve()),
                        'module_sha256':file_sha(__file__),'library_path':str(self.path),
                        'library_sha256':file_sha(self.path),'cufft_version':version.value,
                        'distribution_version':'10.9.0.58','scope':actual,
                        'runtime_files_sha256':entry['runtime_files'],
                        'original_backend':'cuFFT10900 statically linked in original Torch2.0.0+cu118',
                        'original_static_and_standalone_dynamic_bytes_identical_claim':False,
                        'source_and_hyperparameters_changed':False,'selection_uses_seed_or_prediction':False,
                        'calls':self.calls,'call_count':0,'expected_call_count':len(self.expected_batches)*actual['e_layers']}

    def _declare(self):
        I=ctypes.c_int; LL=ctypes.c_longlong; SZ=ctypes.c_size_t; P=ctypes.c_void_p
        signatures = {'cufftGetVersion':[ctypes.POINTER(I)],'cufftCreate':[ctypes.POINTER(I)],
                      'cufftSetAutoAllocation':[I,I],
                      'cufftXtMakePlanMany':[I,I,ctypes.POINTER(LL),ctypes.POINTER(LL),LL,LL,I,ctypes.POINTER(LL),LL,LL,I,LL,ctypes.POINTER(SZ),I],
                      'cufftSetStream':[I,P],'cufftSetWorkArea':[I,P],
                      'cufftXtExec':[I,P,P,I],'cufftDestroy':[I]}
        for name, args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=I

    @staticmethod
    def _check(status, name):
        if status != 0:
            raise RuntimeError(f'{name} returned cuFFT status {status}')

    def __enter__(self):
        self.old_rfft=self.torch.fft.rfft
        self.torch.fft.rfft=self.rfft
        return self

    def __exit__(self, error_type, error, traceback):
        self.torch.fft.rfft=self.old_rfft
        self.receipt['call_count']=len(self.calls)
        self.receipt['original_rfft_restored']=self.torch.fft.rfft is self.old_rfft
        self.receipt['expected_call_count_matched']=len(self.calls)==self.receipt['expected_call_count']
        if error_type is None and not self.receipt['expected_call_count_matched']:
            raise RuntimeError('TimesNet FFT 调用次数与原层数/完整 Test 批次不一致')
        return False

    def rfft(self, x, n=None, dim=-1, norm=None, *, out=None):
        torch=self.torch
        if not (x.is_cuda and x.dtype==torch.float32 and x.ndim==3 and not x.requires_grad):
            raise ValueError('TimesNet 兼容 FFT 仅接受 CUDA FP32 无梯度三维输入')
        if n is not None or dim not in (1,-2) or norm not in (None,'backward') or out is not None:
            raise ValueError('TimesNet 兼容 FFT 实参偏离原 dim=1/n=None/norm=None 调用')
        call_index=len(self.calls); layer_count=self.scope['e_layers']
        if call_index>=len(self.expected_batches)*layer_count:
            raise ValueError('出现超出完整 Test 范围的 FFT 调用')
        expected_batch=self.expected_batches[call_index//layer_count]
        expected_shape=(expected_batch,self.scope['seq_len'],self.scope['d_model'])
        if tuple(x.shape)!=expected_shape:
            raise ValueError('FFT 输入形状/原 batch 或顺序发生变化: '+repr(tuple(x.shape)))
        batch,length,channels=x.shape
        source=x.permute(0,2,1).reshape(batch*channels,length)
        if not source.is_contiguous():
            raise ValueError('FFT 输入不能按原 ATen simple_layout 合并')
        result=torch.empty((batch*channels,length//2+1),dtype=torch.complex64,device=x.device)
        handle=ctypes.c_int();self._check(self.lib.cufftCreate(ctypes.byref(handle)),'cufftCreate')
        stream=torch.cuda.current_stream(x.device)
        try:
            self._check(self.lib.cufftSetAutoAllocation(handle,0),'cufftSetAutoAllocation')
            size=(ctypes.c_longlong*1)(length);workspace_size=ctypes.c_size_t()
            # 与原 ATen CuFFTPlanCache simple_layout 路径一致，null embeds，FP32→complex64。
            self._check(self.lib.cufftXtMakePlanMany(handle,1,size,None,1,1,0,None,1,1,4,
                                                    batch*channels,ctypes.byref(workspace_size),4),'cufftXtMakePlanMany')
            workspace=torch.empty((workspace_size.value,),dtype=torch.uint8,device=x.device)
            self._check(self.lib.cufftSetStream(handle,ctypes.c_void_p(stream.cuda_stream)),'cufftSetStream')
            self._check(self.lib.cufftSetWorkArea(handle,ctypes.c_void_p(workspace.data_ptr())),'cufftSetWorkArea')
            self._check(self.lib.cufftXtExec(handle,ctypes.c_void_p(source.data_ptr()),ctypes.c_void_p(result.data_ptr()),-1),'cufftXtExec')
            stream.synchronize()
            self.calls.append({'input_shape':list(x.shape),'input_stride':list(x.stride()),
                               'plan_input_shape':list(source.shape),'plan_input_stride':list(source.stride()),
                               'length':length,'batch':batch*channels,'workspace_bytes':workspace_size.value,
                               'stream':stream.cuda_stream,'device':str(x.device),'direction':-1,
                               'normalization':'none','input_type':0,'output_type':4,'execution_type':4})
            return result.reshape(batch,channels,length//2+1).permute(0,2,1)
        finally:
            self._check(self.lib.cufftDestroy(handle),'cufftDestroy')
