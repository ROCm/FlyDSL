#!/bin/bash
set -euo pipefail

ACTION=${1:-launch}
MODEL_PATH=${2:-meta-llama/Meta-Llama-3-8B-Instruct}

if [ $# -ge 2 ]; then
  shift 2
else
  shift $#
fi

EXTRA_ARGS=("$@")
ATOM_PORT="${ATOM_SERVER_PORT:-8000}"
ATOM_BASE_URL="http://localhost:${ATOM_PORT}"
ATOM_SERVER_LOG="${ATOM_SERVER_LOG:-/tmp/atom_server.log}"

wait_for_server() {
  local atom_server_pid=$1
  local max_retries=30
  local retry_interval=60
  local server_up=false

  echo "========== Waiting for ATOM server to start =========="
  for ((i=1; i<=max_retries; i++)); do
    if ! kill -0 "${atom_server_pid}" 2>/dev/null; then
      echo "ATOM server process exited unexpectedly."
      echo "Last 50 lines of server log:"
      tail -50 "${ATOM_SERVER_LOG}" 2>/dev/null || true
      exit 1
    fi

    if curl -sf "${ATOM_BASE_URL}/health" -o /dev/null; then
      echo "ATOM server HTTP endpoint is ready."
      server_up=true
      break
    fi

    echo "Waiting for ATOM server... (${i}/${max_retries})"
    sleep "${retry_interval}"
  done

  if [ "${server_up}" = false ]; then
    echo "ATOM server did not become ready in time."
    kill "${atom_server_pid}" || true
    exit 1
  fi
}

warm_up_server() {
  local atom_server_pid=$1
  local warmup_retries=10
  local warmup_interval=30
  local warmup_done=false

  echo "========== Warming up ATOM server =========="
  for ((i=1; i<=warmup_retries; i++)); do
    if ! kill -0 "${atom_server_pid}" 2>/dev/null; then
      echo "ATOM server process exited during warmup."
      echo "Last 50 lines of server log:"
      tail -50 "${ATOM_SERVER_LOG}" 2>/dev/null || true
      exit 1
    fi

    if curl -sf "${ATOM_BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"hi\",\"max_tokens\":1}" \
      -o /dev/null --max-time 120; then
      echo "ATOM server warmup completed."
      warmup_done=true
      break
    fi

    echo "Warmup attempt ${i}/${warmup_retries} failed, retrying in ${warmup_interval}s..."
    sleep "${warmup_interval}"
  done

  if [ "${warmup_done}" = false ]; then
    echo "ATOM server warmup failed."
    kill "${atom_server_pid}" || true
    exit 1
  fi
}

case "${ACTION}" in
  launch)
    echo "========== Launching ATOM server =========="
    pkill -f 'atom.entrypoints' || true
    rm -rf ~/.cache/atom/*

    PROFILER_ARGS=()
    if [ "${ENABLE_TORCH_PROFILER:-0}" = "1" ]; then
      PROFILER_ARGS=(--torch-profiler-dir /app/trace --mark-trace)
      echo "Torch profiler enabled."
    fi

    python3 -m atom.entrypoints.openai_server \
      --model "${MODEL_PATH}" \
      "${PROFILER_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" 2>&1 | tee "${ATOM_SERVER_LOG}" &
    atom_server_pid=$!

    echo "${atom_server_pid}" > /tmp/flydsl_atom_server.pid
    wait_for_server "${atom_server_pid}"
    warm_up_server "${atom_server_pid}"
    ;;

  accuracy)
    echo "========== Running ATOM accuracy test =========="
    if ! command -v lm_eval >/dev/null 2>&1; then
      python3 -m pip install --timeout 60 --retries 10 -U 'lm-eval[api]'
    fi

    umask 0022
    mkdir -p accuracy_test_results
    RESULT_FILENAME="accuracy_test_results/$(date +%Y%m%d%H%M%S).json"

    lm_eval \
      --model local-completions \
      --model_args model="${MODEL_PATH}",base_url="${ATOM_BASE_URL}/v1/completions",num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True \
      --tasks gsm8k \
      --num_fewshot 3 \
      --output_path "${RESULT_FILENAME}"

    echo "Accuracy test results saved to ${RESULT_FILENAME}"
    ;;

  stop)
    echo "========== Stopping ATOM server =========="
    pkill -f 'atom.entrypoints' || true
    sleep 2
    pkill -9 -f 'multiprocessing.spawn' || true
    pkill -9 -f 'multiprocessing.resource_tracker' || true
    rm -f /tmp/flydsl_atom_server.pid

    echo "Waiting for GPU memory to release..."
    for i in $(seq 1 60); do
      if rocm-smi --showmemuse >/tmp/flydsl_rocm_smi.log 2>/dev/null; then
        USED_GPUS=$(grep "VRAM%" /tmp/flydsl_rocm_smi.log | awk '{print $NF}' | awk '$1 > 0' | wc -l)
      else
        USED_GPUS=0
      fi
      if [ "${USED_GPUS}" -eq 0 ]; then
        echo "GPU memory released after ${i}s"
        break
      fi
      if [ "${i}" -eq 60 ]; then
        echo "WARNING: GPU memory still in use after 60s."
      fi
      sleep 1
    done
    ;;

  *)
    echo "Unsupported action: ${ACTION}" >&2
    exit 2
    ;;
esac
