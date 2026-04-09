#!/bin/bash
# Kimi 2.5 TP8 W4A16 groupwise(g=32) MoE GEMM benchmark
# model_dim=7168, inter_dim=256, E=384, topk=8, tokens=128
export HIP_VISIBLE_DEVICES=2
export FLYDSL_RUNTIME_ENABLE_CACHE=0
export PYTHONPATH=./build/python_packages:./
export HSA_TOOLS_LIB=""
export HSA_TOOLS_REPORT_LOAD_FAILURE=0

DUMP_DIR=./dumps/kimi25_w4a16
export FLYDSL_DUMP_IR=1
export FLYDSL_DEBUG_DUMP_ASM=1
export FLYDSL_DUMP_DIR=$DUMP_DIR
rm -rf "$DUMP_DIR"

python -c "
import torch; from tests.kernels.test_moe_gemm import run_moe_stage1, run_moe_stage2
kw=dict(tokens=128,model_dim=7168,inter_dim=256,experts=384,topk=8,in_dtype='int4_bf16',group_size=32,doweight_stage1=False,num_iters=64,num_warmup=5,skip_ref=False)
s1=dict(tile_m=16,tile_n=128,tile_k=128)
s2=dict(tile_m=16,tile_n=128,tile_k=256)
print('=== Stage1 params ==='); [print(f'  {k}={v}') for k,v in {**kw,**s1}.items()]
run_moe_stage1(**kw,**s1)
a2=torch.randn(128,8,256,device='cuda',dtype=torch.bfloat16)*0.1
print('=== Stage2 params ==='); [print(f'  {k}={v}') for k,v in {**kw,**s2}.items()]
run_moe_stage2(**kw,**s2,a2_fp8_in=a2,a2_scale_in=None)
"

# --- Resource & Occupancy Analysis ---
# gfx942/gfx950 (CDNA3): 512 VGPRs/SIMD, 4 SIMDs/CU, total 2048 VGPRs/CU
# Max waves/SIMD = 8, LDS/CU = 65536 bytes, workgroup = 256 threads = 4 waves
analyze_occupancy() {
    local ISA_FILE=$1
    local LABEL=$2
    [ -f "$ISA_FILE" ] || return

    local VGPRS=$(grep '.amdhsa_next_free_vgpr' "$ISA_FILE" | awk '{print $2}')
    local SGPRS=$(grep '.amdhsa_next_free_sgpr' "$ISA_FILE" | awk '{print $2}')
    local ACCUM=$(grep '.amdhsa_accum_offset' "$ISA_FILE" | awk '{print $2}')
    local LDS=$(grep '.amdhsa_group_segment_fixed_size' "$ISA_FILE" | awk '{print $2}')

    # Arch VGPRs: allocated in granules of 8 (gfx9), round up
    local ARCH_VGPRS=$ACCUM
    local ARCH_ALLOC=$(( ((ARCH_VGPRS + 7) / 8) * 8 ))
    # AccVGPRs = total - arch offset
    local ACC_VGPRS=$(( VGPRS - ACCUM ))
    local ACC_ALLOC=$(( ((ACC_VGPRS + 3) / 4) * 4 ))
    # Total allocated VGPRs (unified register file view)
    local TOTAL_VGPR_ALLOC=$(( ARCH_ALLOC > (ACCUM + ACC_ALLOC) ? ARCH_ALLOC : (ACCUM + ACC_ALLOC) ))
    # SGPR allocation: granule of 8, +2 for VCC
    local SGPR_ALLOC=$(( ((SGPRS + 2 + 7) / 8) * 8 ))

    # Waves limited by VGPRs: 512 / ceil(total_vgprs/8)*8 per SIMD, but simpler:
    # Each SIMD has 512 VGPRs. waves_vgpr = floor(512 / alloc_per_wave)
    local WAVES_VGPR=$(( 512 / TOTAL_VGPR_ALLOC ))
    [ $WAVES_VGPR -gt 8 ] && WAVES_VGPR=8

    # Waves limited by LDS: 65536 / lds_per_workgroup, * waves_per_wg (4 for 256 threads)
    local WAVES_PER_WG=4
    local WAVES_LDS=8
    if [ "$LDS" -gt 0 ]; then
        # LDS alloc granule = 256 bytes
        local LDS_ALLOC=$(( ((LDS + 255) / 256) * 256 ))
        local WGS_PER_CU=$(( 65536 / LDS_ALLOC ))
        # waves per CU from LDS, then per SIMD (4 SIMDs)
        local WAVES_CU_LDS=$(( WGS_PER_CU * WAVES_PER_WG ))
        WAVES_LDS=$(( WAVES_CU_LDS / 4 ))
        [ $WAVES_LDS -gt 8 ] && WAVES_LDS=8
    fi

    # Effective occupancy
    local WAVES=$(( WAVES_VGPR < WAVES_LDS ? WAVES_VGPR : WAVES_LDS ))
    local OCC_PCT=$(( WAVES * 100 / 8 ))

    echo ""
    echo "===== $LABEL Resource Usage & Occupancy ====="
    printf "  %-24s %s\n" "Arch VGPRs (used/alloc):" "${ARCH_VGPRS} / ${ARCH_ALLOC}"
    printf "  %-24s %s\n" "Acc  VGPRs (used/alloc):" "${ACC_VGPRS} / ${ACC_ALLOC}"
    printf "  %-24s %s\n" "Total VGPRs allocated:"   "${TOTAL_VGPR_ALLOC}"
    printf "  %-24s %s\n" "SGPRs (used/alloc):"      "${SGPRS} / ${SGPR_ALLOC}"
    printf "  %-24s %s bytes" "LDS:"                  "${LDS}"
    if [ "$LDS" -gt 0 ]; then
        printf " (%.1f KB)\n" "$(awk "BEGIN{printf \"%.1f\", $LDS/1024}")"
    else
        printf "\n"
    fi
    printf "  %-24s %d (vgpr-limited=%d, lds-limited=%d)\n" \
        "Waves/SIMD:" "$WAVES" "$WAVES_VGPR" "$WAVES_LDS"
    printf "  %-24s %d%%  (%d/8 waves)\n" "Occupancy:" "$OCC_PCT" "$WAVES"
}

S1_ISA=$DUMP_DIR/moe_gemm1_0/16_final_isa.s
S2_ISA=$DUMP_DIR/moe_gemm2_0/16_final_isa.s

analyze_occupancy "$S1_ISA" "Stage1"
analyze_occupancy "$S2_ISA" "Stage2"

# Dump stage1 hot loop (.LBB0_2)
S1_HOTLOOP=$DUMP_DIR/moe_gemm1_0/hotloop.s
if [ -f "$S1_ISA" ]; then
    sed -n '/^\.LBB0_2:/,/s_cbranch_.*\.LBB0_2/p' "$S1_ISA" > "$S1_HOTLOOP"
    echo ""
    echo "===== Stage1 hot loop (.LBB0_2) -> $S1_HOTLOOP ====="
    echo "  Lines:               $(wc -l < "$S1_HOTLOOP")"
    echo "  buffer_load_dwordx4: $(grep -c 'buffer_load_dwordx4' "$S1_HOTLOOP")"
    echo "  buffer_load_dword:   $(grep -c 'buffer_load_dword ' "$S1_HOTLOOP")"
    echo "  ds_read:             $(grep -c 'ds_read' "$S1_HOTLOOP")"
    echo "  ds_write:            $(grep -c 'ds_write' "$S1_HOTLOOP")"
    echo "  v_mfma:              $(grep -c 'v_mfma' "$S1_HOTLOOP")"
    echo "  s_waitcnt:           $(grep -c 's_waitcnt' "$S1_HOTLOOP")"
    echo ""
    echo "===== Stage1 hot loop instruction mix ====="
    grep -v '^\s*$\|^\s*;\|^\s*//\|^\.LBB\|^\s*\.p2align' "$S1_HOTLOOP" | \
        sed 's/^\s*//; s/\s.*//' | sort | uniq -c | sort -rn
fi

