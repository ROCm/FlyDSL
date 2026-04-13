---
name: build-rocm-image
description: Connect to a remote host via SSH and build a Docker image with rocprofv3, vllm, aiter, FlyDSL, and custom triton (rocm-maxnreg-support-v35 branch). Use when user wants to build/rebuild the ROCm development image on a remote host. Usage: /build-rocm-image <hostname>
allowed-tools: Bash
---

# Build ROCm Development Image

Build a Docker image on a remote host with rocm gpu access based on `rocm/vllm-dev:nightly` with custom triton from the `rocm-maxnreg-support-v35` branch.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<HOST>` | Yes | The remote hostname to SSH into and build the image on. Example: `hjbog-srdc-39.amd.com` |

When this skill is invoked, the argument passed in is the target hostname. Replace all occurrences of `<HOST>` below with the provided hostname. If no hostname is provided, ask the user for it before proceeding.

## Target Host

- **Host**: `<HOST>` (provided as argument)
- **Access**: SSH (key-based authentication)

## Base Image

- **Image**: `rocm/vllm-dev:nightly`
- **Included**: rocprofv3 (ROCm 7.0), PyTorch 2.9

## Customization

- **Triton**: Replace stock triton 3.4.0 with custom build from https://<GITHUB_TOKEN>@github.com/ROCm/triton branch `rocm-maxnreg-support-v35`
- **vLLM**: Replace pre-installed version with https://<GITHUB_TOKEN>@github.com/ROCm/vllm branch `ps_pa`
- **aiter**: Replace pre-installed version with latest from https://<GITHUB_TOKEN>@github.com/ROCm/aiter main branch (develop)
- **FlyDSL**: Install from https://<GITHUB_TOKEN>@github.com/ROCm/FlyDSL develop branch

## Build Steps

### Step 1: Generate Dockerfile on remote host

```bash
ssh -o ConnectTimeout=30 <HOST> "cat > /tmp/Dockerfile.rocm-custom << 'DOCKERFILE'
FROM rocm/vllm-dev:nightly

# Uninstall existing triton, vllm, aiter
RUN pip uninstall -y triton pytorch-triton-rocm vllm aiter 2>/dev/null; true

# Install build dependencies
RUN pip install ninja cmake pybind11

# Clone and build custom triton from ROCm fork (rocm-maxnreg-support-v35 branch)
RUN cd /tmp && \
    git clone --depth 1 --branch rocm-maxnreg-support-v35 https://<GITHUB_TOKEN>@github.com/ROCm/triton.git triton-custom && \
    cd triton-custom/python && \
    pip install -e . && \
    cd / && rm -rf /tmp/triton-custom

# Clone and install aiter from main branch (develop)
RUN cd /tmp && \
    git clone --depth 1 --branch main https://<GITHUB_TOKEN>@github.com/ROCm/aiter.git && \
    cd aiter && \
    pip install -e . && \
    cd / && rm -rf /tmp/aiter

# Clone and install FlyDSL from develop branch
RUN cd /tmp && \
    git clone --depth 1 --branch develop https://<GITHUB_TOKEN>@github.com/ROCm/FlyDSL.git && \
    cd FlyDSL && \
    pip install -e . && \
    cd / && rm -rf /tmp/FlyDSL

# Clone and install vllm from ROCm/vllm ps_pa branch
RUN cd /tmp && \
    git clone --depth 1 --branch ps_pa https://<GITHUB_TOKEN>@github.com/ROCm/vllm.git && \
    cd vllm && \
    pip install -e . && \
    cd / && rm -rf /tmp/vllm

# Install rocprof-trace-decoder: download installer, extract .so, copy to /opt/rocm/lib
RUN cd /tmp && \
    wget -q https://<GITHUB_TOKEN>@github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh && \
    chmod +x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh && \
    ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=/tmp/rtd-install && \
    find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \; && \
    ldconfig && \
    rm -rf /tmp/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh /tmp/rtd-install

# Verify installations
RUN python3 -c 'import triton; print(f\"triton version: {triton.__version__}\")' && \
    python3 -c 'import vllm; print(f\"vllm version: {vllm.__version__}\")' && \
    python3 -c 'import aiter; print(\"aiter OK\")' && \
    python3 -c 'import flydsl; print(\"FlyDSL OK\")' && \
    which rocprofv3 && echo 'rocprofv3 OK' && \
    ls /opt/rocm/lib/librocprof*decoder* && echo 'rocprof-trace-decoder OK'

LABEL description=\"ROCm dev image with vllm(ROCm/ps_pa), aiter(main), FlyDSL(develop), rocprofv3, rocprof-trace-decoder, and custom triton (rocm-maxnreg-support-v35)\"
DOCKERFILE
"
```

### Step 2: Build the image

Build the image with a descriptive tag. Use `--network=host` to ensure git clone works.

```bash
ssh -o ConnectTimeout=30 <HOST> "docker build --network=host -t rocm-dev-custom:triton-maxnreg-v35 -f /tmp/Dockerfile.rocm-custom /tmp"
```

**Note**: The triton build can take 30-60 minutes. Use `--progress=plain` to see full build logs.

### Step 3: Verify the built image

```bash
ssh -o ConnectTimeout=30 <HOST> "docker run --rm rocm-dev-custom:triton-maxnreg-v35 bash -c '
echo \"=== Triton ===\"
python3 -c \"import triton; print(triton.__version__)\"
echo \"=== vLLM ===\"
python3 -c \"import vllm; print(vllm.__version__)\"
echo \"=== aiter ===\"
python3 -c \"import aiter; print(aiter.__version__)\" 2>/dev/null || python3 -c \"import aiter; print(\\\"aiter OK\\\")\"
echo \"=== FlyDSL ===\"
python3 -c \"import flydsl; print(flydsl.__version__)\" 2>/dev/null || python3 -c \"import flydsl; print(\\\"FlyDSL OK\\\")\"
echo \"=== rocprofv3 ===\"
rocprofv3 --version 2>/dev/null || which rocprofv3
echo \"=== ROCm ===\"
cat /opt/rocm/.info/version
'"
```

### Step 4: Clean up

```bash
ssh -o ConnectTimeout=30 <HOST> "rm -f /tmp/Dockerfile.rocm-custom"
```

## Output

Report to the user:
- The image name and tag
- Versions of triton, vllm, aiter, and ROCm inside the image
- The triton git branch used
- Any build warnings or errors

## Error Handling

- If SSH connection fails, inform the user they need a valid SSH key and Conductor reservation
- If triton build fails, check if `rocm-maxnreg-support-v35` branch exists and suggest verifying the branch name
- If disk space is insufficient, suggest cleaning unused images with `docker image prune`

## Example Usage

To start a container from the built image with GPU access:

```bash
ssh <HOST> "docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=64g rocm-dev-custom:triton-maxnreg-v35 bash"
```
