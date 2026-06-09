  /opt/venv/bin/python profile_flydsl_16384.py \
    --mode graph \
    --runners aiter_mxfp4_moe,local_mxfp4_all_flydsl \
    --warmup 20 \
    --iters 10 \
    --graph-iters 40 \
    --repeat 5 \
    --expected-kernels 8 \
    --trace-dir profiler_traces
