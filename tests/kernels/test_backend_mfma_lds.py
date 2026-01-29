#!/usr/bin/env python3
"""
Backend Correctness Tests for MFMA and LDS operations.

These tests use the debug kernels from kernels/debug_*.py to verify
ASM backend correctness for advanced operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import struct
from flydsl.runtime.device import get_rocm_arch


def u32_to_f32(u):
    """Convert unsigned 32-bit int to float."""
    return struct.unpack('f', struct.pack('I', u & 0xFFFFFFFF))[0]


def compare_i32_outputs(ee_out, asm_out, name="", tolerance=0):
    """Compare i32 outputs from two backends."""
    ee_list = ee_out.cpu().tolist()
    asm_list = asm_out.cpu().tolist()
    
    if tolerance == 0:
        match = ee_list == asm_list
    else:
        # Allow small differences for floating point results stored as i32
        match = True
        for i, (e, a) in enumerate(zip(ee_list, asm_list)):
            e_f, a_f = u32_to_f32(e), u32_to_f32(a)
            if abs(e_f - a_f) > tolerance:
                match = False
                break
    
    if not match:
        print(f"\n{name} MISMATCH:")
        print(f"  EE:  {ee_list[:16]}...")
        print(f"  ASM: {asm_list[:16]}...")
    
    return match


# =============================================================================
# Test: Single MFMA
# =============================================================================

class TestSingleMFMA:
    """Test single MFMA instruction."""
    
    def test_single_mfma_fp8(self):
        """Test single fp8 MFMA instruction."""
        from kernels.debug_single_mfma import compile_single_mfma_test
        
        # Clear cache to ensure fresh compilation
        compile_single_mfma_test.cache_clear()
        
        # Input: 8 bytes of FP8 ~1.0 (0x3c)
        a_bytes = bytes([0x3c] * 8)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 8, 4)]
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        
        # Compile and run both backends
        try:
            ee = compile_single_mfma_test(backend='execution_engine')
            compile_single_mfma_test.cache_clear()
            asm = compile_single_mfma_test(backend='asm')
            
            out_ee = torch.zeros(6, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(6, device='cuda', dtype=torch.int32)
            
            ee(a_in, out_ee)
            asm(a_in, out_asm)
            torch.cuda.synchronize()
            
            # Compare loaded data (slots 4-5)
            load_match = compare_i32_outputs(out_ee[4:6], out_asm[4:6], "single_mfma_load")
            assert load_match, "Single MFMA: input load mismatch"
            
            # Compare MFMA result (slots 0-3), allow small fp tolerance
            result_match = compare_i32_outputs(out_ee[:4], out_asm[:4], "single_mfma_result", tolerance=1e-5)
            assert result_match, "Single MFMA: result mismatch"
            
            print("  single_mfma_fp8: PASS")
        except Exception as e:
            pytest.fail(f"single_mfma_fp8 failed: {e}")


# =============================================================================
# Test: MFMA Unit Test
# =============================================================================

class TestMFMAUnit:
    """Test MFMA unit with separate A and B inputs."""
    
    def test_mfma_unit(self):
        """Test MFMA with two fp8 MFMAs chained."""
        from kernels.debug_mfma_unit import compile_mfma_unit_test
        
        compile_mfma_unit_test.cache_clear()
        
        # 16 bytes each for A and B (4 dwords each)
        a_bytes = bytes([0x3c] * 16)  # FP8 ~1.0
        b_bytes = bytes([0x3c] * 16)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 16, 4)]
        b_dwords = [int.from_bytes(b_bytes[i:i+4], 'little') for i in range(0, 16, 4)]
        
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        b_in = torch.tensor(b_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_mfma_unit_test(backend='execution_engine')
            compile_mfma_unit_test.cache_clear()
            asm = compile_mfma_unit_test(backend='asm')
            
            out_ee = torch.zeros(4, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(4, device='cuda', dtype=torch.int32)
            
            ee(a_in, b_in, out_ee)
            asm(a_in, b_in, out_asm)
            torch.cuda.synchronize()
            
            result_match = compare_i32_outputs(out_ee, out_asm, "mfma_unit", tolerance=1e-5)
            assert result_match, "MFMA unit: result mismatch"
            
            print("  mfma_unit: PASS")
        except Exception as e:
            pytest.fail(f"mfma_unit failed: {e}")


# =============================================================================
# Test: MFMA Pressure Test
# =============================================================================

class TestMFMAPressure:
    """Test MFMA with high VGPR pressure."""
    
    @pytest.mark.parametrize("pressure", [16, 32, 64])
    def test_mfma_pressure(self, pressure):
        """Test MFMA with varying VGPR pressure levels."""
        from kernels.debug_fp8_mfma_pressure_test import compile_fp8_mfma_pressure_unit
        
        compile_fp8_mfma_pressure_unit.cache_clear()
        
        # 8 dwords for input packs
        pack_bytes = bytes([0x3c] * 32)
        pack_dwords = [int.from_bytes(pack_bytes[i:i+4], 'little') for i in range(0, 32, 4)]
        packs_in = torch.tensor(pack_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_fp8_mfma_pressure_unit(pressure_i32=pressure, backend='execution_engine')
            compile_fp8_mfma_pressure_unit.cache_clear()
            asm = compile_fp8_mfma_pressure_unit(pressure_i32=pressure, backend='asm')
            
            # Output: 4 dwords for MFMA result + pressure_i32 for temps
            out_size = 4 + pressure
            out_ee = torch.zeros(out_size, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(out_size, device='cuda', dtype=torch.int32)
            
            ee(out_ee, packs_in)
            asm(out_asm, packs_in)
            torch.cuda.synchronize()
            
            # Compare MFMA results
            mfma_match = compare_i32_outputs(out_ee[:4], out_asm[:4], f"mfma_pressure_{pressure}", tolerance=1e-5)
            assert mfma_match, f"MFMA pressure (p={pressure}): MFMA result mismatch"
            
            # Compare temps (should be exact)
            temps_match = compare_i32_outputs(out_ee[4:], out_asm[4:], f"mfma_pressure_temps_{pressure}")
            assert temps_match, f"MFMA pressure (p={pressure}): temps mismatch"
            
            print(f"  mfma_pressure p={pressure}: PASS")
        except Exception as e:
            pytest.fail(f"mfma_pressure p={pressure} failed: {e}")


# =============================================================================
# Test: LDS Only
# =============================================================================

class TestLDSOnly:
    """Test LDS read/write operations."""
    
    def test_lds_simple(self):
        """Test simple LDS write then read."""
        from kernels.debug_lds_only import compile_debug_lds_only
        
        compile_debug_lds_only.cache_clear()
        
        # Input: sequential bytes for easy verification
        tile_m, tile_k = 16, 512
        total_bytes = tile_m * tile_k
        a_data = bytes(range(256)) * (total_bytes // 256)
        a_in = torch.frombuffer(bytearray(a_data), dtype=torch.int8).cuda()
        
        total_threads = 256
        dbg_size = total_threads * 5  # 5 i32s per thread
        
        try:
            ee = compile_debug_lds_only(backend='execution_engine')
            compile_debug_lds_only.cache_clear()
            asm = compile_debug_lds_only(backend='asm')
            
            dbg_ee = torch.zeros(dbg_size, device='cuda', dtype=torch.int32)
            dbg_asm = torch.zeros(dbg_size, device='cuda', dtype=torch.int32)
            
            ee(dbg_ee, a_in)
            asm(dbg_asm, a_in)
            torch.cuda.synchronize()
            
            # Compare all debug outputs
            match = compare_i32_outputs(dbg_ee, dbg_asm, "lds_simple")
            assert match, "LDS simple: output mismatch"
            
            print("  lds_simple: PASS")
        except Exception as e:
            pytest.fail(f"lds_simple failed: {e}")


# =============================================================================
# Test: LDS with XOR16 Swizzle
# =============================================================================

class TestLDSSwizzle:
    """Test LDS with XOR16 swizzle pattern."""
    
    def test_lds_xor16(self):
        """Test LDS XOR16 swizzle pattern used in preshuffle GEMM."""
        from kernels.debug_lds_xor16 import compile_debug_lds_xor16
        
        compile_debug_lds_xor16.cache_clear()
        
        # Input data
        tile_m, tile_k = 16, 512
        total_bytes = tile_m * tile_k
        a_data = bytes(range(256)) * (total_bytes // 256)
        a_in = torch.frombuffer(bytearray(a_data), dtype=torch.int8).cuda()
        
        total_threads = 256
        dbg_size = total_threads * 8  # 8 i32s per thread (4 for A, 4 for B)
        
        try:
            ee = compile_debug_lds_xor16(backend='execution_engine')
            compile_debug_lds_xor16.cache_clear()
            asm = compile_debug_lds_xor16(backend='asm')
            
            dbg_ee = torch.zeros(dbg_size, device='cuda', dtype=torch.int32)
            dbg_asm = torch.zeros(dbg_size, device='cuda', dtype=torch.int32)
            
            ee(dbg_ee, a_in)
            asm(dbg_asm, a_in)
            torch.cuda.synchronize()
            
            # Compare only A data (slots 0-3 for each thread)
            # B data (slots 4-7) has known issues in the debug kernel itself
            ee_a = dbg_ee.view(total_threads, 8)[:, :4].flatten()
            asm_a = dbg_asm.view(total_threads, 8)[:, :4].flatten()
            
            match = compare_i32_outputs(ee_a, asm_a, "lds_xor16_a_data")
            assert match, "LDS XOR16: A data output mismatch"
            
            print("  lds_xor16: PASS")
        except Exception as e:
            pytest.fail(f"lds_xor16 failed: {e}")


# =============================================================================
# Test: MFMA Data Types (INT8, FP16, BF16, FP8)
# =============================================================================

class TestMFMADataTypes:
    """Test MFMA with different data types."""
    
    def test_mfma_i8(self):
        """Test INT8 MFMA (mfma_i32_16x16x32_i8)."""
        from kernels.debug_mfma_dtypes import compile_mfma_i8_test
        
        compile_mfma_i8_test.cache_clear()
        
        # Input: 8 bytes of i8 value 1
        a_bytes = bytes([1] * 8)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 8, 4)]
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_mfma_i8_test(backend='execution_engine')
            compile_mfma_i8_test.cache_clear()
            asm = compile_mfma_i8_test(backend='asm')
            
            out_ee = torch.zeros(4, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(4, device='cuda', dtype=torch.int32)
            
            ee(a_in, out_ee)
            asm(a_in, out_asm)
            torch.cuda.synchronize()
            
            match = compare_i32_outputs(out_ee, out_asm, "mfma_i8")
            assert match, "MFMA INT8: result mismatch"
            
            print("  mfma_i8: PASS")
        except Exception as e:
            pytest.fail(f"mfma_i8 failed: {e}")
    
    def test_mfma_f16(self):
        """Test FP16 MFMA (mfma_f32_16x16x16f16)."""
        from kernels.debug_mfma_dtypes import compile_mfma_f16_test
        
        compile_mfma_f16_test.cache_clear()
        
        # Input: 4 fp16 values of 1.0 (0x3c00)
        a_bytes = bytes([0x00, 0x3c] * 4)  # fp16 1.0 = 0x3c00 (little endian)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 8, 4)]
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_mfma_f16_test(backend='execution_engine')
            compile_mfma_f16_test.cache_clear()
            asm = compile_mfma_f16_test(backend='asm')
            
            out_ee = torch.zeros(4, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(4, device='cuda', dtype=torch.int32)
            
            ee(a_in, out_ee)
            asm(a_in, out_asm)
            torch.cuda.synchronize()
            
            match = compare_i32_outputs(out_ee, out_asm, "mfma_f16", tolerance=1e-5)
            assert match, "MFMA FP16: result mismatch"
            
            print("  mfma_f16: PASS")
        except Exception as e:
            pytest.fail(f"mfma_f16 failed: {e}")
    
    def test_mfma_bf16(self):
        """Test BF16 MFMA (mfma_f32_16x16x16bf16_1k)."""
        from kernels.debug_mfma_dtypes import compile_mfma_bf16_test
        
        compile_mfma_bf16_test.cache_clear()
        
        # Input: 4 bf16 values of 1.0 (0x3f80)
        a_bytes = bytes([0x80, 0x3f] * 4)  # bf16 1.0 = 0x3f80 (little endian)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 8, 4)]
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_mfma_bf16_test(backend='execution_engine')
            compile_mfma_bf16_test.cache_clear()
            asm = compile_mfma_bf16_test(backend='asm')
            
            out_ee = torch.zeros(4, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(4, device='cuda', dtype=torch.int32)
            
            ee(a_in, out_ee)
            asm(a_in, out_asm)
            torch.cuda.synchronize()
            
            match = compare_i32_outputs(out_ee, out_asm, "mfma_bf16", tolerance=1e-5)
            assert match, "MFMA BF16: result mismatch"
            
            print("  mfma_bf16: PASS")
        except AttributeError as e:
            if "mfma_f32_16x16x16bf16_1k" in str(e):
                pytest.skip("BF16 MFMA not available on this target")
            raise
        except Exception as e:
            pytest.fail(f"mfma_bf16 failed: {e}")
    
    def test_mfma_fp8(self):
        """Test FP8 MFMA (mfma_f32_16x16x32_fp8_fp8)."""
        from kernels.debug_mfma_dtypes import compile_mfma_fp8_test
        
        compile_mfma_fp8_test.cache_clear()
        
        # Input: 8 fp8 values of ~1.0 (0x3c)
        a_bytes = bytes([0x3c] * 8)
        a_dwords = [int.from_bytes(a_bytes[i:i+4], 'little') for i in range(0, 8, 4)]
        a_in = torch.tensor(a_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_mfma_fp8_test(backend='execution_engine')
            compile_mfma_fp8_test.cache_clear()
            asm = compile_mfma_fp8_test(backend='asm')
            
            out_ee = torch.zeros(4, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(4, device='cuda', dtype=torch.int32)
            
            ee(a_in, out_ee)
            asm(a_in, out_asm)
            torch.cuda.synchronize()
            
            match = compare_i32_outputs(out_ee, out_asm, "mfma_fp8", tolerance=1e-5)
            assert match, "MFMA FP8: result mismatch"
            
            print("  mfma_fp8: PASS")
        except Exception as e:
            pytest.fail(f"mfma_fp8 failed: {e}")


# =============================================================================
# Test: FP8 MFMA Unit Test
# =============================================================================

class TestFP8MFMAUnit:
    """Test FP8 MFMA unit operations."""
    
    def test_fp8_mfma_unit(self):
        """Test FP8 MFMA with known inputs."""
        try:
            from kernels.debug_fp8_mfma_unit_test import compile_fp8_mfma_unit_test
        except ImportError:
            pytest.skip("debug_fp8_mfma_unit_test not available")
        
        compile_fp8_mfma_unit_test.cache_clear()
        
        # FP8 inputs
        pack_bytes = bytes([0x3c] * 32)
        pack_dwords = [int.from_bytes(pack_bytes[i:i+4], 'little') for i in range(0, 32, 4)]
        packs_in = torch.tensor(pack_dwords, device='cuda', dtype=torch.int32)
        
        try:
            ee = compile_fp8_mfma_unit_test(backend='execution_engine')
            compile_fp8_mfma_unit_test.cache_clear()
            asm = compile_fp8_mfma_unit_test(backend='asm')
            
            out_ee = torch.zeros(256, device='cuda', dtype=torch.int32)
            out_asm = torch.zeros(256, device='cuda', dtype=torch.int32)
            
            ee(out_ee, packs_in)
            asm(out_asm, packs_in)
            torch.cuda.synchronize()
            
            match = compare_i32_outputs(out_ee, out_asm, "fp8_mfma_unit", tolerance=1e-5)
            assert match, "FP8 MFMA unit: result mismatch"
            
            print("  fp8_mfma_unit: PASS")
        except Exception as e:
            pytest.fail(f"fp8_mfma_unit failed: {e}")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
