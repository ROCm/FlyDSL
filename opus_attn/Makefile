OPUS_INCLUDE_DIR ?= /home/carhuang/repo/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++20 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math -fPIC

BUILD = build
OUT   = $(BUILD)/gqa_attn.exe
PY_SO = $(BUILD)/libopus_gqa_attn.so

KERNEL_SRCS = gqa_d128_kernel_causal.cc gqa_d128_kernel_noncausal.cc \
              gqa_d512_kernel_causal.cc gqa_d512_kernel_noncausal.cc
HOST_SRC    = gqa_host.cc
PY_API_SRC  = gqa_python_api.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gqa_host.o
PY_API_OBJ  = $(BUILD)/gqa_python_api.o

.PHONY: all clean benchmark lib

all: $(OUT)
lib: $(PY_SO)

$(BUILD):
	mkdir -p $(BUILD)

# Kernel objects (heavy — device+host passes, parallelizable via make -j)
# -D__HIPCC_RTC__ skips implicit __clang_hip_runtime_wrapper.h on host pass (~250ms saved)
# Each kernel .cc only includes its own template hpp; depending on both keeps the rule simple.
$(BUILD)/%.o: %.cc gqa_d128_kernel_template.hpp gqa_d512_kernel_template.hpp gqa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

# Host object (no -D__HIPCC_RTC__ — needs full HIP runtime for <<<>>> syntax)
$(HOST_OBJ): $(HOST_SRC) gqa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(PY_API_OBJ): $(PY_API_SRC) gqa_defs.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

# Link
$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

$(PY_SO): $(KERNEL_OBJS) $(PY_API_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -shared -o $@

benchmark: $(OUT)
	@echo "=== d=512 Causal ==="     && $(OUT) -d 512 --causal
	@echo "=== d=512 Non-causal ===" && $(OUT) -d 512 --no-causal
	@echo "=== d=128 Causal ==="     && $(OUT) -d 128 --causal
	@echo "=== d=128 Non-causal ===" && $(OUT) -d 128 --no-causal

clean:
	rm -rf $(BUILD)
