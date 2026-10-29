#!/usr/bin/env bash
# Read-only MegaMoE environment/provenance preflight.
set -u

usage() {
  printf '%s\n' \
    "Usage: $0 [--aiter PATH] [--flydsl PATH] [--runner PATH] [--python PATH]" \
    "Prints repository, import, runtime, GPU, process and runner provenance." \
    "It does not build, launch, kill, clean, reset or modify repositories."
}

aiter_path=""
flydsl_path=""
runner_path=""
python_path="python3"

while (($#)); do
  case "$1" in
    --aiter)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      aiter_path=$2
      shift 2
      ;;
    --flydsl)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      flydsl_path=$2
      shift 2
      ;;
    --runner)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      runner_path=$2
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      python_path=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

section() {
  printf '\n[%s]\n' "$1"
}

repo_info() {
  local label=$1
  local path=$2
  section "$label repository"
  if [[ -z "$path" ]]; then
    printf 'not specified\n'
    return
  fi
  if ! git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'not a git worktree: %s\n' "$path"
    return
  fi
  printf 'path=%s\n' "$(cd "$path" && pwd -P)"
  printf 'head=%s\n' "$(git -C "$path" rev-parse HEAD 2>/dev/null || true)"
  printf 'branch=%s\n' "$(git -C "$path" branch --show-current 2>/dev/null || true)"
  printf 'describe=%s\n' "$(git -C "$path" describe --always --dirty --tags 2>/dev/null || true)"
  printf 'status:\n'
  git -C "$path" status --short --branch 2>&1 || true
  printf 'remotes:\n'
  git -C "$path" remote -v 2>&1 | sed -E 's#(https?://)[^/@]+@#\1<redacted>@#g' || true
}

section "timestamp and host"
printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
uname -a 2>&1 || true

repo_info "AITER" "$aiter_path"
repo_info "FlyDSL" "$flydsl_path"

section "selected environment"
for name in \
  HIP_VISIBLE_DEVICES \
  WORLD_SIZE \
  MASTER_ADDR \
  MASTER_PORT \
  MORI_SHMEM_HEAP_SIZE \
  FLYDSL_RUNTIME_ENABLE_CACHE \
  FLYDSL_RUNTIME_CACHE_DIR \
  FLYDSL_RUNTIME_RUN_ONLY \
  PYTHONPATH \
  LD_LIBRARY_PATH; do
  if [[ -v $name ]]; then
    printf '%s=%s\n' "$name" "${!name}"
  else
    printf '%s=<unset>\n' "$name"
  fi
done

section "Python paths and distributions"
printf 'requested_python=%s\n' "$python_path"
if command -v "$python_path" >/dev/null 2>&1 || [[ -x "$python_path" ]]; then
  "$python_path" - <<'PY' 2>&1 || true
import importlib.metadata
import importlib.util
import platform
import sys

print(f"executable={sys.executable}")
print(f"python={platform.python_version()}")
for name in ("torch", "flydsl", "aiter", "mori"):
    try:
        spec = importlib.util.find_spec(name)
        print(f"{name}.origin={None if spec is None else spec.origin}")
    except Exception as exc:
        print(f"{name}.origin_error={type(exc).__name__}: {exc}")
    try:
        print(f"{name}.distribution_version={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}.distribution_version=<not installed as a distribution>")
PY
else
  printf 'python executable not found\n'
fi

section "GPU inventory"
gpu_count=0
for device_path in /sys/class/drm/card*/device; do
  [[ -r "$device_path/product_name" ]] || continue
  card_path=${device_path%/device}
  card_name=${card_path##*/}
  gpu_count=$((gpu_count + 1))
  printf '%s' "$card_name"
  for field_name in product_name product_number unique_id mem_info_vram_total mem_info_vram_used; do
    if [[ -r "$device_path/$field_name" ]]; then
      IFS= read -r field_value < "$device_path/$field_name" || field_value='<unreadable>'
      printf ' %s=%s' "$field_name" "$field_value"
    fi
  done
  printf '\n'
done
printf 'gpu_count=%s (sysfs only; no GPU runtime initialization)\n' "$gpu_count"

section "/dev/kfd holders"
if [[ -e /dev/kfd ]]; then
  holder_count=0
  for proc_path in /proc/[0-9]*; do
    pid=${proc_path##*/}
    [[ -r "$proc_path/status" ]] || continue
    for fd_path in "$proc_path"/fd/*; do
      fd_target=$(readlink "$fd_path" 2>/dev/null || true)
      [[ "$fd_target" == /dev/kfd ]] || continue
      holder_count=$((holder_count + 1))
      holder_user=$(stat -c '%U' "$proc_path" 2>/dev/null || printf '?')
      holder_comm=$(tr -d '\000' < "$proc_path/comm" 2>/dev/null || printf '?')
      holder_cgroup=$(sed -n '1p' "$proc_path/cgroup" 2>/dev/null || true)
      printf 'pid=%s user=%s comm=%s cgroup=%s\n' "$pid" "$holder_user" "$holder_comm" "$holder_cgroup"
      break
    done
  done
  printf 'holder_count=%s (resolved through /proc/*/fd)\n' "$holder_count"
else
  printf '/dev/kfd unavailable\n'
fi

section "GPU-related processes"
ps -eo user=,pid=,ppid=,etimes=,stat=,comm= 2>&1 \
  | awk 'BEGIN { IGNORECASE=1 } /torchrun|torch\.distributed|test_mega_moe|bench_mega_moe|sglang|atom|vllm|rocprof|rocprofv3/ { print }' \
  || true

section "runner"
if [[ -z "$runner_path" ]]; then
  printf 'not specified\n'
elif [[ ! -f "$runner_path" ]]; then
  printf 'not found: %s\n' "$runner_path"
else
  printf 'path=%s\n' "$(cd "$(dirname "$runner_path")" && pwd -P)/$(basename "$runner_path")"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$runner_path" 2>&1 || true
  fi
  stat -c 'mode=%A owner=%U:%G size=%s mtime=%y' "$runner_path" 2>&1 || true
fi

section "preflight conclusion"
printf '%s\n' \
  'This report is descriptive. Verify process ownership before cleanup and' \
  'record the exact benchmark command, model geometry, route, MTPR and raw logs.'
