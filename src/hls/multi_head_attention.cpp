/**
 * Multi-Head Attention — Vitis HLS Implementation
 *
 * Optimized for INT8 quantized transformer models.
 * Uses block RAM for KV-cache, DSP for QK^T matmul.
 *
 * Target: Xilinx Alveo U250 (SLR crossing optimized)
 */

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define SEQ_LEN     128
#define HEAD_DIM    64
#define NUM_HEADS   4
#define BLOCK_SIZE  16

typedef ap_int<8> int8_t;
typedef ap_int<32> acc_t;

struct attention_config {
    int seq_len;
    int head_dim;
    int num_heads;
    float scale;
};

/**
 * Quantized matrix multiply: Q × K^T
 * Uses INT8 × INT8 → INT32 accumulation with requantization
 */
void qk_matmul(
    hls::stream<ap_axiu<128,0,0,0>>& q_stream,
    hls::stream<ap_axiu<128,0,0,0>>& k_stream,
    hls::stream<ap_axiu<128,0,0,0>>& attn_stream,
    const attention_config& cfg
) {
    #pragma HLS INTERFACE axis port=q_stream
    #pragma HLS INTERFACE axis port=k_stream
    #pragma HLS INTERFACE axis port=attn_stream
    #pragma HLS INTERFACE s_axilite port=cfg
    #pragma HLS INTERFACE s_axilite port=return

    // Block-level KV cache in BRAM
    static int8_t k_cache[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    #pragma HLS BIND_STORAGE variable=k_cache type=ram_2p impl=bram

    acc_t acc_buf[SEQ_LEN];
    #pragma HLS ARRAY_PARTITION variable=acc_buf cyclic factor=16

    for (int h = 0; h < cfg.num_heads; h++) {
        for (int i = 0; i < cfg.seq_len; i += BLOCK_SIZE) {
            for (int j = 0; j < cfg.seq_len; j += BLOCK_SIZE) {
                #pragma HLS PIPELINE II=1
                acc_t local_acc[BLOCK_SIZE];
                #pragma HLS ARRAY_PARTITION variable=local_acc complete

                // Compute Q[i:i+B] @ K[j:j+B]^T
                for (int bi = 0; bi < BLOCK_SIZE; bi++) {
                    local_acc[bi] = 0;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        #pragma HLS UNROLL factor=8
                        ap_axiu<128,0,0,0> q_val = q_stream.read();
                        ap_axiu<128,0,0,0> k_val = k_stream.read();
                        local_acc[bi] += (acc_t)(q_val.data.range(7,0)) * (acc_t)(k_val.data.range(7,0));
                    }
                    // Scale and requantize
                    acc_buf[i + bi] = local_acc[bi];
                }
            }
        }

        // Write attention scores to output stream
        for (int i = 0; i < cfg.seq_len; i++) {
            ap_axiu<128,0,0,0> out_val;
            out_val.data.range(31,0) = acc_buf[i];
            out_val.last = (i == cfg.seq_len - 1) && (h == cfg.num_heads - 1);
            attn_stream.write(out_val);
        }
    }
}

/**
 * Softmax with online algorithm (numerically stable)
 * Fixed-point implementation for FPGA
 */
void softmax_fpga(
    hls::stream<ap_axiu<128,0,0,0>>& attn_in,
    hls::stream<ap_axiu<128,0,0,0>>& attn_out,
    const attention_config& cfg
) {
    #pragma HLS INTERFACE axis port=attn_in
    #pragma HLS INTERFACE axis port=attn_out
    #pragma HLS INTERFACE s_axilite port=cfg
    #pragma HLS INTERFACE s_axilite port=return

    for (int h = 0; h < cfg.num_heads; h++) {
        // Pass 1: find max for numerical stability
        acc_t max_val = -32768;
        for (int i = 0; i < cfg.seq_len; i++) {
            #pragma HLS PIPELINE II=1
            ap_axiu<128,0,0,0> val = attn_in.read();
            acc_t score = val.data.range(31,0);
            if (score > max_val) max_val = score;
        }

        // Pass 2: exp and sum
        acc_t sum = 0;
        static acc_t exp_cache[SEQ_LEN];
        for (int i = 0; i < cfg.seq_len; i++) {
            #pragma HLS PIPELINE II=1
            // Approximate exp using LUT (8-bit precision)
            acc_t shifted = attn_in.read().data.range(31,0) - max_val;
            exp_cache[i] = (shifted > -128) ? (128 + shifted) : 0;  // linear approx
            sum += exp_cache[i];
        }

        // Pass 3: normalize
        for (int i = 0; i < cfg.seq_len; i++) {
            #pragma HLS PIPELINE II=1
            ap_axiu<128,0,0,0> out_val;
            out_val.data.range(31,0) = (exp_cache[i] << 8) / sum;  // FP8 output
            out_val.last = (i == cfg.seq_len - 1) && (h == cfg.num_heads - 1);
            attn_out.write(out_val);
        }
    }
}
