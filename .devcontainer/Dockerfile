# FlyDSL FMHA dev environment — ROCm 7.13 + PyTorch 2.10 (gfx94X / MI308X class GPUs)
FROM registry-sc-harbor.amd.com/framework/therock-main:1192_gfx94X_7.13.0.dev0-2f6ad0ecbd89ed1962d48dadd7ed2ec6b8a5c12f_ubuntu24.04_py3.12_pytorch_release-2.10_2.10.0_40d237bfae8

SHELL ["/bin/bash", "-lc"]

# Extra Python deps for FlyDSL tests/benchmarks
RUN pip install --no-cache-dir \
    pytest \
    pandas \
    ninja \
    cmake \
    pybind11 \
    nanobind

WORKDIR /workspaces/FlyDSL
