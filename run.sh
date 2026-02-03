set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'
alias python='python3'

# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=1
# export HIP_VISIBLE_DEVICES=3
# export HIP_VISIBLE_DEVICES=5
export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7


# export LD_LIBRARY_PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
# export PATH=/mnt/raid0/heyanguang/code/poc_kl/scripts/common:$PATH

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"
pip show triton
rocprofv3 --version


function run_flydsl_op {
    export FLIR_LOG_MORE=1
    export FLIR_DUMP_IR=1
    export FLIR_REBUILD=1
    export FLIR_DUMP_DIR=./flydsl_dump

    # python tests/kernels/test_moe_stage1_simple.py --size M

    python tests/kernels/test_simple_gemm.py --size XL
    # python tests/kernels/test_simple_gemm.py --size NA4
}


function get_flydsl_op_thread_trace {
    pushd $PWD
    export KERNEL_NAME=kernel_gemm
    KERNEL_VERSION="${KERNEL_NAME}_v0"


    DUMP_TRACE=1
    # DUMP_TRACE=0
    if [ $DUMP_TRACE = 1 ]; then
        rm -rf ./pass_2
        cd ./thread_trace
        trace_dir=./${KERNEL_VERSION}
        rm -rf ./rpf_v3
        rm -rf ./${trace_dir} ./${trace_dir}.tar.gz
        mkdir -p ${trace_dir}
        cd -

        rocprofv3 -i ./input.yaml -- \
        python tests/kernels/test_simple_gemm.py --size XL

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
get_flydsl_op_thread_trace


set +x
