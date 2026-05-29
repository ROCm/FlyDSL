from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="opus_asm_cpp",
    ext_modules=[
        CUDAExtension(
            "opus_asm_ext",
            sources=["opus_asm_ext.cc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
            extra_link_args=["-Wl,--no-as-needed"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
