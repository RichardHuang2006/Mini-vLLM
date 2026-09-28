#pragma once

#include <stdexcept>
#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mini_vllm_ext {

void load_library(const char *path);

// Kernels are instantiated per dtype and looked up by name: "rms_norm_bf16" and so on.
inline std::string kernel_name(const char *op, mx::Dtype dtype) {
    if (dtype == mx::float32) return std::string(op) + "_f32";
    if (dtype == mx::float16) return std::string(op) + "_f16";
    return std::string(op) + "_bf16";
}

// Every op is GPU-only and has no vmap, but an MLX primitive must still declare both.
#define MINI_VLLM_PRIMITIVE(Name)                                                                    \
    void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {             \
        throw std::runtime_error(#Name " runs on the GPU only");                                   \
    }                                                                                              \
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override; \
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &,       \
                                                             const std::vector<int> &) override {  \
        throw std::runtime_error(#Name " has no vmap");                                            \
    }                                                                                              \
    const char *name() const override { return #Name; }

// rms_norm.cpp: x [..., D] * rsqrt(mean(x^2) + eps) * weight [D].
mx::array rms_norm(const mx::array &x, const mx::array &weight, float eps, mx::StreamOrDevice s = {});

class RMSNorm : public mx::Primitive {
public:
    RMSNorm(mx::Stream stream, float eps) : mx::Primitive(stream), eps_(eps) {}
    MINI_VLLM_PRIMITIVE(RMSNorm)

private:
    float eps_;
};

// swiglu.cpp: silu(gate) * up, elementwise.
mx::array swiglu(const mx::array &gate, const mx::array &up, mx::StreamOrDevice s = {});

class SwiGLU : public mx::Primitive {
public:
    explicit SwiGLU(mx::Stream stream) : mx::Primitive(stream) {}
    MINI_VLLM_PRIMITIVE(SwiGLU)
};

// rope.cpp: x [B, L, H, D] rotated by the fp32 cos/sin tables at positions [L] or [B, L].
mx::array rope(const mx::array &x, const mx::array &positions, const mx::array &cos, const mx::array &sin,
               mx::StreamOrDevice s = {});

class RoPE : public mx::Primitive {
public:
    explicit RoPE(mx::Stream stream) : mx::Primitive(stream) {}
    MINI_VLLM_PRIMITIVE(RoPE)
};

// paged_attention.cpp: the cache write, in place, and attention over the ragged batch.
mx::array paged_cache_update(const mx::array &pool, const mx::array &slot_mapping, const mx::array &values,
                             mx::StreamOrDevice s = {});

class PagedCacheUpdate : public mx::Primitive {
public:
    explicit PagedCacheUpdate(mx::Stream stream) : mx::Primitive(stream) {}
    MINI_VLLM_PRIMITIVE(PagedCacheUpdate)
};

mx::array paged_attention(const mx::array &q, const mx::array &key_pages, const mx::array &value_pages,
                          const mx::array &block_tables, const mx::array &cu_seqlens_q,
                          const mx::array &context_lens, float scale, float k_scale, float v_scale,
                          mx::StreamOrDevice s = {});

class PagedAttention : public mx::Primitive {
public:
    PagedAttention(mx::Stream stream, float scale, float k_scale, float v_scale)
        : mx::Primitive(stream), scale_(scale), k_scale_(k_scale), v_scale_(v_scale) {}
    MINI_VLLM_PRIMITIVE(PagedAttention)

private:
    float scale_, k_scale_, v_scale_;
};

// kv_quantize.cpp: fp8 quantize and scatter into a uint8 pool, in place, in one pass.
mx::array kv_quantize_scatter(const mx::array &pool, const mx::array &slot_mapping, const mx::array &values,
                              float scale, mx::StreamOrDevice s = {});

class KvQuantizeScatter : public mx::Primitive {
public:
    KvQuantizeScatter(mx::Stream stream, float scale) : mx::Primitive(stream), scale_(scale) {}
    MINI_VLLM_PRIMITIVE(KvQuantizeScatter)

private:
    float scale_;
};

// decode_attention.cpp: dense attention for a few query tokens, q [B, H_q, L, D] against
// k, v [B, H_k, S, D]; causal or unmasked.
mx::array decode_attention(const mx::array &q, const mx::array &k, const mx::array &v, float scale, bool causal,
                           mx::StreamOrDevice s = {});

class DecodeAttention : public mx::Primitive {
public:
    DecodeAttention(mx::Stream stream, float scale, bool causal, int num_heads, int num_kv_heads)
        : mx::Primitive(stream), scale_(scale), causal_(causal), num_heads_(num_heads), num_kv_heads_(num_kv_heads) {}
    MINI_VLLM_PRIMITIVE(DecodeAttention)

private:
    float scale_;
    bool causal_;
    int num_heads_, num_kv_heads_;
};

// flash_prefill.cpp: tiled MMA prefill for bf16, D = 128, in the same dense layout.
mx::array flash_prefill(const mx::array &q, const mx::array &k, const mx::array &v, float scale, bool causal,
                        mx::StreamOrDevice s = {});

class FlashPrefill : public mx::Primitive {
public:
    FlashPrefill(mx::Stream stream, float scale, bool causal, int num_heads, int num_kv_heads)
        : mx::Primitive(stream), scale_(scale), causal_(causal), num_heads_(num_heads), num_kv_heads_(num_kv_heads) {}
    MINI_VLLM_PRIMITIVE(FlashPrefill)

private:
    float scale_;
    bool causal_;
    int num_heads_, num_kv_heads_;
};

}  // namespace mini_vllm_ext
