#!/usr/bin/env python3
"""
CuTe Runtime Python Package Setup
==================================

Build and install the CuTe runtime Python bindings.
Supports AMD ROCm/HIP (default) and NVIDIA CUDA.

Build:
    python setup.py build_ext --inplace

Install:
    pip install .

Development:
    pip install -e .
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(Extension):
    """Custom extension type for CMake-based builds."""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """Custom build_ext command that invokes CMake."""
    
    def run(self):
        # Check CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build CuTe runtime")
        
        # Check ROCm is available (AMD GPU)
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if not os.path.exists(rocm_path):
            print(f"Warning: ROCm not found at {rocm_path}")
            print("Building without GPU runtime support")
        else:
            print(f"Found ROCm at {rocm_path}")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        # ROCm/HIP support for AMD GPUs
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if os.path.exists(rocm_path):
            cmake_args.append('-DENABLE_ROCM=ON')
            cmake_args.append('-DUSE_ROCM=ON')
            cmake_args.append('-DUSE_CUDA=OFF')
            # GFX942 for MI300 series
            hip_arch = os.environ.get('HIP_ARCHITECTURES', 'gfx942')
            cmake_args.append(f'-DHIP_ARCHITECTURES={hip_arch}')
            print(f"Building with ROCm support for {hip_arch}")
        else:
            cmake_args.append('-DENABLE_ROCM=OFF')
            cmake_args.append('-DUSE_ROCM=OFF')
            cmake_args.append('-DBUILD_RUNTIME=OFF')
            print("Building without GPU runtime")
        
        # MLIR path (if available)
        mlir_path = os.environ.get('MLIR_INSTALL_DIR')
        if not mlir_path:
            # Try to find MLIR in common locations
            for path in ['/mnt/raid0/felix/llvm-project/buildmlir',
                        '/usr/local/lib/cmake/mlir',
                        '/usr/lib/llvm-*/lib/cmake/mlir']:
                if os.path.exists(path):
                    mlir_path = path
                    break
        
        if mlir_path:
            cmake_args.append(f'-DMLIR_DIR={mlir_path}/lib/cmake/mlir')
            cmake_args.append(f'-DLLVM_DIR={mlir_path}/lib/cmake/llvm')
            print(f"Using MLIR from {mlir_path}")
        else:
            print("MLIR not found, TableGen targets will be skipped")
        
        # Build configuration
        build_args = ['--config', 'Release']
        build_args += ['--', '-j8']
        
        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Run CMake configure
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        
        # Run CMake build
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return "CuTe Runtime - Python bindings for CuTe MLIR compiler with AMD ROCm support"

setup(
    name='cute-runtime',
    version='0.1.0',
    author='CuTe IR Project',
    description='Python runtime library for CuTe MLIR compiler (AMD ROCm/HIP)',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/cute-ir-tablegen',
    
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    
    ext_modules=[CMakeExtension('cute_runtime._cute_bindings')],
    cmdclass={'build_ext': CMakeBuild},
    
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'mypy',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme',
        ],
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Compilers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
    ],
    
    zip_safe=False,
)
