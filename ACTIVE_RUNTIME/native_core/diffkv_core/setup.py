import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='diffkv_core',
    ext_modules=[
        CUDAExtension(
            name='diffkv_core',
            sources=[
                'src/compressor_thread.cpp',
                'src/paging_stream.cu',
                'src/bindings.cpp'
            ],
            include_dirs=[os.path.join(this_dir, 'include')],
            libraries=['cusolver', 'cublas'],
            extra_compile_args={'cxx': ['/O2'], 'nvcc': ['-O3', '-allow-unsupported-compiler']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
