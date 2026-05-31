#!/bin/bash
# Build FPGA-ML-Inference
# Usage: ./scripts/build.sh [hw|sw_emu|hw_emu]

set -e

TARGET=${1:-hw}
PLATFORM="xilinx_u250_gen3x16_xdma_4_1_202210_1"

echo "🔧 Building FPGA-ML-Inference (target: $TARGET)"

# Check Vitis
if [ -z "$XILINX_VITIS" ]; then
    echo "⚠️  Sourcing Vitis 2023.2..."
    source /opt/xilinx/Vitis/2023.2/settings64.sh
fi

# Check ROCm
if [ -z "$ROCM_PATH" ]; then
    export ROCM_PATH=/opt/rocm
fi

# Build HLS kernels
echo "📦 Compiling HLS kernels..."
mkdir -p build
for kernel in src/hls/*.cpp; do
    name=$(basename $kernel .cpp)
    echo "  → $name"
    v++ -c -t $TARGET --platform $PLATFORM --kernel $name -o build/${name}.xo $kernel
done

# Link kernels
echo "🔗 Linking design..."
v++ -l -t $TARGET --platform $PLATFORM     build/multi_head_attention.xo     build/gelu_activation.xo     -o build/fpga_ml.xclbin

# Build host
echo "🖥️  Building host application..."
mkdir -p bin
hipcc -O2 -std=c++17     -I./src -I$ROCM_PATH/include -I/opt/xilinx/xrt/include     -o bin/benchmark     src/host/main.cpp     -L$ROCM_PATH/lib -L/opt/xilinx/xrt/lib     -lamdhip64 -lxrt_coreutil -lOpenCL

echo "✅ Build complete!"
echo "   Binary: bin/benchmark"
echo "   XCLBIN: build/fpga_ml.xclbin"
