---
name: build-rocm-image
description: Connect to a remote host via SSH and build a Docker image with rocprofv3, aiter, and FlyDSL. Use when user wants to build/rebuild the ROCm development image on a remote host. Usage: /build-rocm-image <hostname>
allowed-tools: Bash
---

# Build ROCm Development Image

Build a Docker image on a remote host with rocm gpu access based on `rocm/vllm-dev:nightly`.

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

- **aiter**: Replace pre-installed version with latest from https://github.com/ROCm/aiter main branch
- **FlyDSL**: Install from https://github.com/ROCm/FlyDSL main branch
- **rocprof-trace-decoder**: Install the release that matches the ROCm version in the base image

## Build Steps

### Step 1: Generate Dockerfile on remote host

```bash
ssh -o ConnectTimeout=30 <HOST> "cat > /tmp/Dockerfile.rocm-custom << 'DOCKERFILE'
FROM rocm/vllm-dev:nightly

# Uninstall existing aiter
RUN pip uninstall -y aiter 2>/dev/null; true

# Install build dependencies
RUN pip install ninja cmake pybind11

# Clone and install aiter from main branch
RUN cd /tmp && \
    git clone --depth 1 --branch main https://github.com/ROCm/aiter.git && \
    cd aiter && \
    pip install -e . && \
    cd / && rm -rf /tmp/aiter

# Clone and install FlyDSL from main branch
RUN cd /tmp && \
    git clone --depth 1 --branch main https://github.com/ROCm/FlyDSL.git && \
    cd FlyDSL && \
    pip install -e . && \
    cd / && rm -rf /tmp/FlyDSL

# Install the rocprof-trace-decoder release that matches the ROCm version.
RUN set -eux; \
    ROCM_VERSION="$(sed -E 's/^([0-9]+)\.([0-9]+).*/\1.\2/' /opt/rocm/.info/version)"; \
    RTD_VERSION="0.1.5"; \
    echo "Using rocprof-trace-decoder ${RTD_VERSION} for ROCm ${ROCM_VERSION}"; \
    RTD_INSTALLER="rocprof-trace-decoder-manylinux-2.28-${RTD_VERSION}-Linux.sh"; \
    cd /tmp; \
    wget -q "https://github.com/ROCm/rocprof-trace-decoder/releases/download/${RTD_VERSION}/${RTD_INSTALLER}"; \
    chmod +x "${RTD_INSTALLER}"; \
    "./${RTD_INSTALLER}" --skip-license --prefix=/tmp/rtd-install; \
    find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \; ; \
    ldconfig; \
    rm -rf "${RTD_INSTALLER}" /tmp/rtd-install

# Verify installations
RUN python3 -c 'import aiter; print(\"aiter OK\")' && \
    python3 -c 'import flydsl; print(\"FlyDSL OK\")' && \
    which rocprofv3 && echo 'rocprofv3 OK' && \
    ls /opt/rocm/lib/librocprof*decoder* && echo 'rocprof-trace-decoder OK'

LABEL description=\"ROCm dev image with aiter(main), FlyDSL(main), rocprofv3, and rocprof-trace-decoder\"
DOCKERFILE
"
```

### Step 2: Build the image

Build the image with a descriptive tag. Use `--network=host` to ensure git clone works.

```bash
ssh -o ConnectTimeout=30 <HOST> "docker build --network=host -t rocm-dev-custom:main -f /tmp/Dockerfile.rocm-custom /tmp"
```

**Note**: Use `--progress=plain` to see full build logs.

### Step 3: Verify the built image

```bash
ssh -o ConnectTimeout=30 <HOST> "docker run --rm rocm-dev-custom:main bash -c '
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
- Versions of aiter, FlyDSL, ROCm, and the rocprof-trace-decoder release selected for that ROCm version
- Any build warnings or errors

## Error Handling

- If SSH connection fails, inform the user they need a valid SSH key and Conductor reservation
- If rocprof-trace-decoder mapping fails, check the ROCm version in `/opt/rocm/.info/version` and add the matching decoder release from ROCm release notes
- If disk space is insufficient, suggest cleaning unused images with `docker image prune`

## Example Usage

To start a container from the built image with GPU access:

```bash
ssh <HOST> "docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size=64g rocm-dev-custom:main bash"
```
