"""Setup script for FLIR Python bindings.

NOTE: Prefer installing from repo root:
  - Editable: `pip install -e .` (or `python setup.py develop`)
  - Wheel:    `python -m build` (or `python setup.py bdist_wheel`)

The root package includes the embedded `_mlir` runtime from
`build/python_packages/pyflir/_mlir`, which this `python/setup.py` does not
package by itself.
"""

import os
from setuptools import setup, find_packages

# Read requirements
def load_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt") as f:
            requirements = [
                line.strip() for line in f 
                if line.strip() 
                and not line.startswith("#") 
                and not line.startswith("--")  # Skip pip options like --index-url
            ]
    return requirements

setup(
    name="pyflir",
    version="0.1.0",
    description="Python bindings for FLIR - ROCm Domain Specific Language for layout algebra",
    long_description=open("../README.md").read() if os.path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="FLIR Contributors",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=load_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering",
    ],
    keywords="mlir cuda rocm gpu compiler layout",
    project_urls={
        "Source": "https://github.com/yourusername/flir",
    },
)
