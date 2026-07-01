#!/usr/bin/env bash
# megaMoE 合并验收一键脚本 (branch: universe_ghu_tmp)
#
# 用法:
#   bash scripts/verify_megamoe_merge.sh                 # 自动检测 arch,跑适用档位
#   BASELINE_CSV=/path/dev.csv bash scripts/verify_megamoe_merge.sh   # 附带 async benchmark 基线对比
#   RUN_EP8=1 bash scripts/verify_megamoe_merge.sh       # 额外跑多卡 EP8 冒烟(需 >=8 卡 + mori 运行时)
#
# 退出码: 0 全部通过(或合理 skip); 非 0 有硬失败。

set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

export PYTHONPATH="${REPO}/build-fly/python_packages:${REPO}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${REPO}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"

FAIL=0
step() { echo; echo "==================== $* ===================="; }
run()  { echo "+ $*"; "$@"; local rc=$?; if [ $rc -ne 0 ]; then echo "!! FAILED (rc=$rc): $*"; FAIL=1; fi; return $rc; }
soft() { echo "+ $* (non-fatal)"; "$@" || echo "~~ non-fatal failure, continuing: $*"; }

# ---- 检测 arch ----
ARCH="$(python -c 'from flydsl.runtime.device import get_rocm_arch; print(get_rocm_arch())' 2>/dev/null)"
if [ -z "$ARCH" ]; then echo "无法检测 GPU arch,退出"; exit 2; fi
echo "检测到 GPU arch: $ARCH"

# ============================================================
step "档位 0 — 基础冒烟 (任意 arch, 必做)"
# ============================================================
run python -c "from flydsl.expr.meta import dsl_loc_tracing; from flydsl.expr.extern import ExternFunction; import mori.ir.flydsl; print('core+extern+mori.ir.flydsl OK')"
run python -c "from flydsl.expr import buffer_ops; assert hasattr(buffer_ops,'extract_aligned_ptr') and hasattr(buffer_ops,'extract_workgroup_aligned_ptr'); from flydsl.expr.rocdl import mfma_i32_16x16x64_i8; print('backfill funcs OK')"

echo "+ dev async 符号未回退检查 (§4)"
N=$(grep -c "use_async_copy\|dma_x_tile_to_lds\|hot_loop_scheduler" kernels/moe_gemm_2stage.py)
if [ "$N" -gt 0 ]; then echo "  async 符号命中 $N 处 OK"; else echo "!! async 符号丢失 (回退风险)"; FAIL=1; fi

echo "+ s_setprio 保持注释检查 (§4.5)"
if grep -q "# rocdl.s_setprio(1)" kernels/mixed_moe_gemm_2stage.py; then echo "  s_setprio 注释态 OK"; else echo "!! s_setprio 未保持注释"; FAIL=1; fi

# ============================================================
step "档位 1 — 功能测试"
# ============================================================
run    pytest tests/kernels/test_moe_reduce.py tests/kernels/test_a8w4_moe_gemm_2stage.py -v
run    pytest tests/kernels/test_pa.py tests/kernels/test_pa_swa.py -v
# test_moe_gemm 依赖 aiter,环境异常时不算硬失败
soft   pytest tests/kernels/test_moe_gemm.py -v

# ============================================================
case "$ARCH" in
  gfx95*)
    step "档位 2 — gfx950/MI355X (async 路径)"
    run pytest tests/kernels/test_moe_gemm_mxscale_gfx1250.py -v
    if [ -n "${BASELINE_CSV:-}" ] && [ -f "${BASELINE_CSV}" ]; then
      echo "+ async benchmark 对比基线 ${BASELINE_CSV}"
      FLYDSL_MOE_USE_ASYNC_COPY=1 bash scripts/run_benchmark.sh --only moe --output_csv /tmp/megamoe_cur.csv
      run python3 scripts/compare_benchmark.py "${BASELINE_CSV}" /tmp/megamoe_cur.csv
    else
      echo "~~ 未提供 BASELINE_CSV,跳过 async benchmark 对比"
      echo "   (async 回退不报错只掉性能,生产强烈建议: BASELINE_CSV=dev.csv 重跑本脚本)"
    fi
    ;;
  gfx1250)
    step "档位 3 — gfx1250/MI450 (回退簇 + wmma)"
    run python -c "import kernels.moe_blockscale_2stage, kernels.gemm_fp8fp4_gfx1250, kernels.preshuffle_gemm; print('gfx1250 revert cluster import OK')"
    run pytest tests/kernels/test_moe_gemm_wmma_gfx1250.py -v
    ;;
  gfx942)
    echo; echo "== arch=gfx942: async(档位2)/gfx1250(档位3) 不适用,已跳过 =="
    ;;
  *)
    echo; echo "== arch=$ARCH: 无对应专项档位,仅跑通用档位 =="
    ;;
esac

# ============================================================
if [ "${RUN_EP8:-0}" = "1" ]; then
  step "档位 4 — 多卡 EP8 megaMoE 冒烟"
  NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  if [ "${NGPU:-0}" -ge 8 ]; then
    run pytest tests/kernels/test_profiler_moe_gemm2_combine.py -v
  else
    echo "~~ 仅 ${NGPU} 卡 (<8),跳过 EP8"
  fi
fi

# ============================================================
step "结果"
if [ "$FAIL" -eq 0 ]; then echo "✅ 全部通过 (arch=$ARCH)"; else echo "❌ 存在硬失败,见上文 !! 行"; fi
exit $FAIL
