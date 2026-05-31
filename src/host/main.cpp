/**
 * FPGA-ML-Inference Host Application
 *
 * Manages GPU-FPGA pipeline:
 * 1. Load model weights to GPU and FPGA DDR
 * 2. Run preprocessing on CPU
 * 3. Run embedding on GPU (ROCm/HIP)
 * 4. Run attention on FPGA (XRT)
 * 5. Run FFN on GPU
 * 6. Collect results
 *
 * Target: AMD GPU + Xilinx Alveo U250
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <fstream>
#include <hip/hip_runtime.h>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "model_loader.h"
#include "pipeline.h"

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

struct PipelineConfig {
    std::string model_path;
    std::string input_path;
    std::string xclbin_path;
    int batch_size;
    int seq_len;
    int hidden_dim;
    int num_heads;
    int num_layers;
    bool use_fpga;
    bool profile;
};

struct BenchmarkResult {
    double preprocess_ms;
    double embedding_ms;
    double attention_ms;
    double ffn_ms;
    double total_ms;
    double throughput_inf_per_sec;
};

/**
 * FPGA attention kernel wrapper
 */
class FPGAAttention {
    xrt::device device;
    xrt::kernel attn_kernel;
    xrt::kernel softmax_kernel;
    xrt::bo q_buf, k_buf, v_buf, out_buf;

public:
    FPGAAttention(const std::string& xclbin_path, int seq_len, int head_dim) {
        device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbin_path);
        attn_kernel = xrt::kernel(uuid, "qk_matmul");
        softmax_kernel = xrt::kernel(uuid, "softmax_fpga");

        size_t buf_size = seq_len * head_dim * sizeof(int8_t);
        q_buf = xrt::bo(device, buf_size, attn_kernel.group_id(0));
        k_buf = xrt::bo(device, buf_size, attn_kernel.group_id(1));
        v_buf = xrt::bo(device, buf_size, attn_kernel.group_id(2));
        out_buf = xrt::bo(device, buf_size, attn_kernel.group_id(3));
    }

    void run(const int8_t* Q, const int8_t* K, const int8_t* V, float* output) {
        q_buf.write(Q);
        k_buf.write(K);
        v_buf.write(V);

        auto run = attn_kernel(q_buf, k_buf, out_buf);
        run.wait();

        softmax_kernel(out_buf, out_buf).wait();
        out_buf.read(output);
    }
};

/**
 * GPU FFN kernel (ROCm/HIP)
 */
__global__ void ffn_forward_gpu(
    const float* __restrict__ input,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    float* __restrict__ output,
    int hidden_dim, int ffn_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    float sum = 0.0f;
    for (int j = 0; j < ffn_dim; j++) {
        float h = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            h += input[k] * W1[k * ffn_dim + j];
        }
        // GELU activation
        float gelu = 0.5f * h * (1.0f + tanhf(0.7978845608f * (h + 0.044715f * h * h * h)));
        sum += gelu * W2[j * hidden_dim + idx];
    }
    output[idx] = sum;
}

BenchmarkResult run_pipeline(const PipelineConfig& cfg) {
    BenchmarkResult result = {};
    auto t_start = std::chrono::high_resolution_clock::now();

    // Load model
    ModelLoader model(cfg.model_path);
    auto t_load = std::chrono::high_resolution_clock::now();

    // Allocate GPU memory
    float *d_input, *d_embed, *d_output;
    CHECK_HIP(hipMalloc(&d_input, cfg.seq_len * cfg.hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_embed, cfg.seq_len * cfg.hidden_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_output, cfg.seq_len * cfg.hidden_dim * sizeof(float)));

    // Load input
    std::ifstream fin(cfg.input_path, std::ios::binary);
    std::vector<float> input(cfg.seq_len * cfg.hidden_dim);
    fin.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
    CHECK_HIP(hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));

    // FPGA attention (if available)
    FPGAAttention* fpga_attn = nullptr;
    if (cfg.use_fpga) {
        fpga_attn = new FPGAAttention(cfg.xclbin_path, cfg.seq_len, cfg.hidden_dim / cfg.num_heads);
    }

    auto t_prep = std::chrono::high_resolution_clock::now();
    result.preprocess_ms = std::chrono::duration<double, std::milli>(t_prep - t_load).count();

    // Embedding (GPU)
    int threads = 256;
    int blocks = (cfg.seq_len * cfg.hidden_dim + threads - 1) / threads;
    hipLaunchKernelGGL(ffn_forward_gpu, dim3(blocks), dim3(threads), 0, 0,
        d_input, model.get_embed_weight(), model.get_embed_output(),
        d_embed, cfg.hidden_dim, cfg.hidden_dim);
    CHECK_HIP(hipDeviceSynchronize());

    auto t_embed = std::chrono::high_resolution_clock::now();
    result.embedding_ms = std::chrono::duration<double, std::milli>(t_embed - t_prep).count();

    // Attention (FPGA or GPU fallback)
    auto t_attn_start = std::chrono::high_resolution_clock::now();
    if (fpga_attn) {
        std::vector<int8_t> Q(cfg.seq_len * cfg.hidden_dim);
        std::vector<int8_t> K(cfg.seq_len * cfg.hidden_dim);
        std::vector<int8_t> V(cfg.seq_len * cfg.hidden_dim);
        std::vector<float> attn_out(cfg.seq_len * cfg.hidden_dim);
        CHECK_HIP(hipMemcpy(Q.data(), d_embed, Q.size(), hipMemcpyDeviceToHost));
        memcpy(K.data(), Q.data(), K.size());
        memcpy(V.data(), Q.data(), V.size());
        fpga_attn->run(Q.data(), K.data(), V.data(), attn_out.data());
        CHECK_HIP(hipMemcpy(d_embed, attn_out.data(), attn_out.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    auto t_attn_end = std::chrono::high_resolution_clock::now();
    result.attention_ms = std::chrono::duration<double, std::milli>(t_attn_end - t_attn_start).count();

    // FFN (GPU)
    hipLaunchKernelGGL(ffn_forward_gpu, dim3(blocks), dim3(threads), 0, 0,
        d_embed, model.get_ffn_weight1(), model.get_ffn_weight2(),
        d_output, cfg.hidden_dim, cfg.hidden_dim * 4);
    CHECK_HIP(hipDeviceSynchronize());

    auto t_ffn = std::chrono::high_resolution_clock::now();
    result.ffn_ms = std::chrono::duration<double, std::milli>(t_ffn - t_attn_end).count();

    auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.throughput_inf_per_sec = 1000.0 / result.total_ms;

    // Cleanup
    hipFree(d_input);
    hipFree(d_embed);
    hipFree(d_output);
    delete fpga_attn;

    return result;
}

int main(int argc, char* argv[]) {
    PipelineConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 128;
    cfg.hidden_dim = 256;
    cfg.num_heads = 4;
    cfg.num_layers = 4;
    cfg.use_fpga = true;
    cfg.profile = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) cfg.model_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0) cfg.input_path = argv[++i];
        else if (strcmp(argv[i], "--xclbin") == 0) cfg.xclbin_path = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0) cfg.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu-only") == 0) cfg.use_fpga = false;
        else if (strcmp(argv[i], "--profile") == 0) cfg.profile = true;
    }

    std::cout << "=== FPGA-ML-Inference Benchmark ===" << std::endl;
    std::cout << "Model: " << cfg.model_path << std::endl;
    std::cout << "FPGA: " << (cfg.use_fpga ? "enabled" : "disabled") << std::endl;

    auto result = run_pipeline(cfg);

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Preprocess:  " << result.preprocess_ms << " ms" << std::endl;
    std::cout << "Embedding:   " << result.embedding_ms << " ms" << std::endl;
    std::cout << "Attention:   " << result.attention_ms << " ms" << std::endl;
    std::cout << "FFN:         " << result.ffn_ms << " ms" << std::endl;
    std::cout << "Total:       " << result.total_ms << " ms" << std::endl;
    std::cout << "Throughput:  " << result.throughput_inf_per_sec << " inf/s" << std::endl;

    return 0;
}
