#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

class ModelLoader {
    std::vector<float> embed_weight_;
    std::vector<float> embed_output_;
    std::vector<float> ffn_weight1_;
    std::vector<float> ffn_weight2_;

public:
    ModelLoader(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open model: " + path);

        auto read_vec = [&](std::vector<float>& v) {
            uint32_t size;
            f.read(reinterpret_cast<char*>(&size), 4);
            v.resize(size);
            f.read(reinterpret_cast<char*>(v.data()), size * sizeof(float));
        };

        read_vec(embed_weight_);
        read_vec(embed_output_);
        read_vec(ffn_weight1_);
        read_vec(ffn_weight2_);
    }

    const float* get_embed_weight() const { return embed_weight_.data(); }
    const float* get_embed_output() const { return embed_output_.data(); }
    const float* get_ffn_weight1() const { return ffn_weight1_.data(); }
    const float* get_ffn_weight2() const { return ffn_weight2_.data(); }
};
