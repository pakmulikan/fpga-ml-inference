# fpga-ml-inference

FPGA-accelerated machine learning inference engine for transformer models. Optimized for AMD/Xilinx FPGAs with Vitis HLS and ROCm integration.

## Overview

This project implements a heterogeneous computing pipeline that offloads compute-intensive transformer attention layers to FPGA fabric while running embedding and output layers on AMD GPUs via ROCm. The goal is to achieve sub-millisecond inference latency for real-time signal processing in biomedical applications.

## Architecture

```
Input Signal → Preprocessing (CPU) → Embedding (GPU/ROCm) → Attention (FPGA) → FFN (GPU) → Output
```

### Key Components

- **HLS Kernels**: Vitis HLS implementations of multi-head attention, layer normalization, and GELU activation
- **Host Code**: ROCm/HIP host application managing GPU-FPGA data transfer via XRT
- **Model Converter**: ONNX → FPGA-optimized quantized model (INT8/FP16)
- **Benchmark Suite**: Latency and throughput comparison across CPU, GPU, and FPGA

## Hardware Requirements

- AMD/Xilinx Alveo U250 or U280 (or Vitis-compatible FPGA)
- AMD GPU with ROCm 6.0+ (tested on MI210, RX 7900 XTX)
- Xilinx Vitis 2023.2+
- XRT (Xilinx Runtime) 2.16+

## Quick Start

```bash
# Setup environment
source /opt/xilinx/Vitis/2023.2/settings64.sh
source /opt/rocm/bin/rocm-setup.sh

# Build HLS kernels
make kernels

# Build host application
make host

# Run benchmark
./bin/benchmark --model models/bert-tiny-int8.xmodel --input data/ecg-sample.bin
```

## Benchmark Results

| Platform | Model | Batch | Latency (ms) | Throughput (inf/s) |
|----------|-------|-------|--------------|-------------------|
| CPU (i7-12700K) | BERT-tiny INT8 | 1 | 12.4 | 80.6 |
| GPU (RX 7900 XTX) | BERT-tiny INT8 | 1 | 2.1 | 476.2 |
| FPGA (U250) | BERT-tiny INT8 | 1 | 0.8 | 1250.0 |
| FPGA+GPU | BERT-tiny INT8 | 1 | 0.5 | 2000.0 |

## Project Structure

```
├── src/
│   ├── hls/          # Vitis HLS kernel sources
│   ├── host/         # ROCm/HIP host application
│   └── kernels/      # XRT kernel wrappers
├── models/           # Pre-converted FPGA models
├── benchmarks/       # Benchmark scripts and results
├── config/           # Hardware and build configurations
├── scripts/          # Build and deployment scripts
├── tests/            # Unit and integration tests
└── docs/             # Architecture documentation
```

## Citation

```bibtex
@software{fpga_ml_inference,
  title = {FPGA-Accelerated ML Inference for Biomedical Signal Processing},
  author = {Rizky Prasetyo},
  year = {2026},
  url = {https://github.com/pakmulikan/fpga-ml-inference}
}
```

## License

MIT License
