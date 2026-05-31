#pragma once
#include <vector>
#include <chrono>

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

template<typename T>
void quantize_fp32_to_int8(const T* input, int8_t* output, int n, float scale) {
    for (int i = 0; i < n; i++) {
        float val = static_cast<float>(input[i]) * scale;
        val = std::max(-128.0f, std::min(127.0f, val));
        output[i] = static_cast<int8_t>(val);
    }
}

template<typename T>
void dequantize_int8_to_fp32(const int8_t* input, T* output, int n, float scale) {
    for (int i = 0; i < n; i++) {
        output[i] = static_cast<T>(static_cast<float>(input[i]) / scale);
    }
}
