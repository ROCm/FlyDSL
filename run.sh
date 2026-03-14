set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'
alias python='python3'

# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=3
# export HIP_VISIBLE_DEVICES=4
export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7

MLIR_LIBS_DIR="$(cd "$(dirname "$0")" && pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs"
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# export LD_LIBRARY_PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
# export PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$PATH

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"
pip show triton
rocprofv3 --version


function run_flydsl_op {
    # export MLIR_ASM_VERBOSE=1
    # export FLYDSL_POSTPROCESS_ISA=./scripts/postprocess_v_interleave.sh
    # export FLYDSL_POSTPROCESS_ISA=./scripts/postprocess_pipeline.sh
    # export FLYDSL_POSTPROCESS_ISA=./scripts/postprocess_vcc_to_sgpr.sh
    export FLIR_LOG_MORE=1
    export FLYDSL_DUMP_IR=1
    export FLYDSL_RUNTIME_ENABLE_CACHE=1
    export FLYDSL_DUMP_DIR=./flydsl_dump

    # python tests/kernels/test_moe_stage1_simple.py --size M

    # python tests/kernels/test_simple_gemm.py --size XL --waves_per_eu 1
    # python tests/kernels/test_simple_gemm.py --size NA4
    # python tests/kernels/test_simple_gemm.py --size all --dtype all

    # python tests/kernels/test_flash_attention_v4_2.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100
    # python tests/kernels/test_flash_attention_v4_3.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --compare-v42
    # python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --compare-v43
    # python tests/kernels/test_flash_attn_func.py --iters 100 --compare-v43

    python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100
    # python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --iters 100
    # python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 256 --head_dim 128 --iters 100

    # python tests/kernels/test_flash_attn_func.py --iters 100

    # python test_mfma_16x16x16_bf16_accvgpr_load.py
    # python test_mfma_16x16x16_bf16_d_vgpr.py
    # python test_buffer_load_to_lds.py
    # python test_qk_gemm_pa.py

    # python fmha_fwd/fmha_fwd_qk_gemm1.py
    # python fmha_fwd/fmha_fwd_pv_gemm2.py
    # python fmha_fwd/fmha_fwd_online_softmax.py
    # python fmha_fwd/fmha_fwd_e2e_single_tile.py
    # python fmha_fwd/fmha_fwd_e2e_single_tile.py --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --iters 6
    # python3 fmha_fwd/fmha_fwd_e2e_full_kernel.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward
    # python3 fmha_fwd/fmha_fwd_e2e_full_kernel.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode backward
    # python3 fmha_fwd/fmha_fwd_e2e_full_kernel.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 1 --iters 100 --warmup 10 --pass1-kv-mode forward
    # python3 fmha_fwd/fmha_fwd_e2e_full_kernel.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 1 --iters 100 --warmup 10 --pass1-kv-mode backward

    # python3 fmha_fwd/fmha_fwd_e2e_v0.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward
    # python3 fmha_fwd/fmha_fwd_e2e_v0.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode backward
    # python3 fmha_fwd/fmha_fwd_e2e_v0.py --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward

    # python3 fmha_fwd/fmha_fwd_e2e_original.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward
    # python3 fmha_fwd/fmha_fwd_e2e_original.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode backward
    # python3 fmha_fwd/fmha_fwd_e2e_original.py --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward
    # python3 fmha_fwd/fmha_fwd_e2e_original.py --batch 1 --num_heads 64 --seq_len 256 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward

    # python3 test_triton_fused_attention.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype bf16
    # python3 test_triton_fused_attention.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype fp16

    # python3 test_triton_flash_attention_mengfei.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype bf16
    # python3 test_triton_flash_attention_mengfei.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype fp16

    # python3 llama_fa_hip/test_llama_fa.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype bf16
    # python3 llama_fa_hip/test_llama_fa.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --warmup 10 --dtype fp16


    # rocprof -i perf_counters1.txt -o prof_v44_p1.csv python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 5 --warmup 2
    # rocprof -i perf_counters2.txt -o prof_v44_p2.csv python tests/kernels/test_flash_attn_func.py --batch 1 --num_heads 64 --seq_len 8192 --head_dim 128 --iters 5 --warmup 2

}


function get_flydsl_op_thread_trace {
    pushd $PWD
    # export KERNEL_NAME=kernel_gemm
    export KERNEL_NAME=fmha_fwd_e2e_full_asm_pipeline_kernel
    KERNEL_VERSION="${KERNEL_NAME}_v0"


    DUMP_TRACE=1
    # DUMP_TRACE=0
    if [ $DUMP_TRACE = 1 ]; then
        mkdir -p ./thread_trace
        rm -rf ./pass_2
        cd ./thread_trace
        trace_dir=./${KERNEL_VERSION}
        rm -rf ./rpf_v3
        rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        mkdir -p ${trace_dir}
        cd -

        rocprofv3 -i ./input.yaml -- \
        python3 fmha_fwd/fmha_fwd_e2e_original.py --batch 1 --num_heads 64 --seq_len 512 --head_dim 128 --tile-opt 0 --iters 100 --warmup 10 --pass1-kv-mode forward
        # python tests/kernels/test_simple_gemm.py --size XL --waves_per_eu 1
        # python tests/kernels/test_simple_gemm.py --size XL

        cd ./thread_trace
        cp -r ./rpf_v3/pass_1/*.att ${trace_dir}
        cp -r ./rpf_v3/pass_1/ui_* ${trace_dir}
        cp -r ./rpf_v3/pass_1/*_agent_info.csv ${trace_dir}
        cp -r ./rpf_v3/pass_1/stats_ui_*.csv ${trace_dir}
        tar -zcf ./${trace_dir}.tar.gz ./${trace_dir}
        ls -lah ./${trace_dir} ./${trace_dir}.tar.gz
        cd -
    fi

    popd
}


# # Press y then n while install
# ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --prefix=/opt/rocm/
# cd /opt/rocm/
# ll -ah ./opt/rocm/lib/librocprof-trace-decoder.so
# ll -ah ./lib/librocprof-trace-decoder.so
# cp opt/rocm/lib/librocprof-trace-decoder.so ./lib/
# ll -ah ./lib/librocprof-trace-decoder.so


run_flydsl_op
# get_flydsl_op_thread_trace


set +x
