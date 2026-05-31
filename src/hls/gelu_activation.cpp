/**
 * GELU Activation — Vitis HLS
 *
 * Piecewise linear approximation for FPGA:
 *   GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
 *
 * Uses 4-segment linear approximation with BRAM LUT.
 */

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define LUT_SIZE 1024
#define LUT_BITS 10

typedef ap_fixed<16,8> fixed_t;

static fixed_t gelu_lut[LUT_SIZE];

/**
 * Initialize GELU LUT (called once at startup)
 */
void init_gelu_lut() {
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = (float)(i - LUT_SIZE/2) / (float)(LUT_SIZE/4);
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        gelu_lut[i] = (fixed_t)gelu;
    }
}

/**
 * GELU forward pass — streaming implementation
 * Processes 16 elements per cycle (II=1 with full unroll)
 */
void gelu_forward(
    hls::stream<ap_axiu<256,0,0,0>>& data_in,
    hls::stream<ap_axiu<256,0,0,0>>& data_out,
    int num_elements
) {
    #pragma HLS INTERFACE axis port=data_in
    #pragma HLS INTERFACE axis port=data_out
    #pragma HLS INTERFACE s_axilite port=num_elements
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS BIND_STORAGE variable=gelu_lut type=rom_1p impl=bram

    for (int i = 0; i < num_elements; i += 16) {
        #pragma HLS PIPELINE II=1
        ap_axiu<256,0,0,0> in_pkt = data_in.read();
        ap_axiu<256,0,0,0> out_pkt;

        for (int j = 0; j < 16; j++) {
            #pragma HLS UNROLL
            fixed_t x;
            x.range(15,0) = in_pkt.data.range(j*16+15, j*16);

            // Quantize to LUT index
            int idx = (int)(x * (LUT_SIZE/4)) + LUT_SIZE/2;
            if (idx < 0) idx = 0;
            if (idx >= LUT_SIZE) idx = LUT_SIZE - 1;

            out_pkt.data.range(j*16+15, j*16) = gelu_lut[idx].range(15,0);
        }

        out_pkt.last = (i + 16 >= num_elements);
        data_out.write(out_pkt);
    }
}
