#!/usr/bin/env python3
"""Tests for INT8 quantization accuracy."""

import sys
sys.path.insert(0, 'src/kernels')

import numpy as np
from model_converter import quantize_tensor


def test_quantize_roundtrip():
    """Verify quantize → dequantize preserves values within tolerance."""
    original = np.random.randn(256).astype(np.float32) * 10
    quantized, scale = quantize_tensor(original)
    dequantized = quantized.astype(np.float32) / scale

    mse = np.mean((original - dequantized) ** 2)
    max_err = np.max(np.abs(original - dequantized))

    assert mse < 0.1, f"MSE too high: {mse}"
    assert max_err < 1.0, f"Max error too high: {max_err}"
    print(f"✅ Roundtrip: MSE={mse:.4f}, max_err={max_err:.4f}")


def test_quantize_range():
    """Verify quantized values are in INT8 range."""
    extreme = np.array([-1000, -100, 0, 100, 1000], dtype=np.float32)
    quantized, scale = quantize_tensor(extreme)

    assert quantized.min() >= -128
    assert quantized.max() <= 127
    print(f"✅ Range: [{quantized.min()}, {quantized.max()}]")


def test_quantize_scale():
    """Verify scale factor is correct."""
    tensor = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    _, scale = quantize_tensor(tensor)

    assert scale > 0
    assert abs(scale - 127.0) < 1.0  # max value maps to 127
    print(f"✅ Scale: {scale:.2f}")


if __name__ == "__main__":
    test_quantize_roundtrip()
    test_quantize_range()
    test_quantize_scale()
    print("\n🎉 All tests passed!")
