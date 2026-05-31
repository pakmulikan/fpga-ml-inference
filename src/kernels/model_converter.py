#!/usr/bin/env python3
"""
ONNX → FPGA-optimized model converter.

Converts ONNX transformer models to INT8 quantized format
suitable for FPGA inference with Vitis HLS kernels.

Usage:
    python model_converter.py --input model.onnx --output model.xmodel --quantize int8
"""

import argparse
import struct
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("pip install onnx")
    exit(1)


def load_onnx_model(path):
    """Load ONNX model and extract weights."""
    model = onnx.load(path)
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)
    return model, weights


def quantize_tensor(tensor, bits=8):
    """Symmetric INT8 quantization with per-tensor scale."""
    if tensor.dtype == np.float32:
        abs_max = np.max(np.abs(tensor))
        if abs_max == 0:
            scale = 1.0
        else:
            scale = (2**(bits-1) - 1) / abs_max
        quantized = np.clip(np.round(tensor * scale), -(2**(bits-1)), 2**(bits-1) - 1).astype(np.int8)
        return quantized, scale
    return tensor, 1.0


def export_fpga_model(weights, output_path, quantize="int8"):
    """Export weights in FPGA-compatible binary format."""
    with open(output_path, 'wb') as f:
        # Header: magic + version + num_tensors
        f.write(b'FPGA')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(weights)))

        for name, tensor in weights.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            if quantize == "int8":
                q_tensor, scale = quantize_tensor(tensor)
                f.write(struct.pack('<f', scale))
                f.write(struct.pack('<I', q_tensor.nbytes))
                f.write(q_tensor.tobytes())
            else:
                f.write(struct.pack('<f', 1.0))
                f.write(struct.pack('<I', tensor.nbytes))
                f.write(tensor.astype(np.float32).tobytes())

    print(f"✅ Exported {len(weights)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX to FPGA model converter")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output .xmodel path")
    parser.add_argument("--quantize", choices=["int8", "fp16", "fp32"], default="int8")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    model, weights = load_onnx_model(args.input)
    print(f"Found {len(weights)} weight tensors")

    export_fpga_model(weights, args.output, args.quantize)


if __name__ == "__main__":
    main()
