# FPGA-ML-Inference Build System
# Requires: Vitis 2023.2+, ROCm 6.0+, XRT 2.16+

VITIS_HOME ?= /opt/xilinx/Vitis/2023.2
ROCM_HOME  ?= /opt/rocm
XRT_HOME   ?= /opt/xilinx/xrt

PLATFORM   ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
TARGET     ?= hw

CXX        := g++
HIPCC      := $(ROCM_HOME)/bin/hipcc
VPP        := $(VITIS_HOME)/bin/v++

CXXFLAGS   := -O2 -std=c++17 -I./src -I$(ROCM_HOME)/include -I$(XRT_HOME)/include
LDFLAGS    := -L$(ROCM_HOME)/lib -L$(XRT_HOME)/lib -lamdhip64 -lxrt_coreutil -lOpenCL

HLS_SRC    := $(wildcard src/hls/*.cpp)
HOST_SRC   := $(wildcard src/host/*.cpp)
KERNEL_SRC := $(wildcard src/kernels/*.cpp)

.PHONY: all kernels host clean benchmark

all: kernels host

kernels: $(patsubst src/hls/%.cpp, build/%.xo, $(HLS_SRC))
	@echo "✅ HLS kernels compiled"

build/%.xo: src/hls/%.cpp
	@mkdir -p build
	$(VPP) -c -t $(TARGET) --platform $(PLATFORM) --kernel $* -o $@ $<

host: $(HOST_SRC)
	@mkdir -p bin
	$(HIPCC) $(CXXFLAGS) -o bin/benchmark $^ $(LDFLAGS)
	@echo "✅ Host application built"

benchmark: host
	./bin/benchmark --model models/bert-tiny-int8.xmodel --input data/ecg-sample.bin --report benchmarks/results.json

clean:
	rm -rf build/ bin/ *.jou *.log *.str
