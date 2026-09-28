#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mini_vllm_ext.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Mini-vLLM's Metal kernels for MLX";

    m.def("load_library", &mini_vllm_ext::load_library, "path"_a);
    m.def("rms_norm", &mini_vllm_ext::rms_norm, "x"_a, "weight"_a, "eps"_a, "stream"_a = nb::none());
    m.def("swiglu", &mini_vllm_ext::swiglu, "gate"_a, "up"_a, "stream"_a = nb::none());
    m.def("rope", &mini_vllm_ext::rope, "x"_a, "positions"_a, "cos"_a, "sin"_a, "stream"_a = nb::none());
    m.def("paged_cache_update", &mini_vllm_ext::paged_cache_update, "pool"_a, "slot_mapping"_a, "values"_a,
          "stream"_a = nb::none());
    m.def("paged_attention", &mini_vllm_ext::paged_attention, "q"_a, "key_pages"_a, "value_pages"_a,
          "block_tables"_a, "cu_seqlens_q"_a, "context_lens"_a, "scale"_a, "k_scale"_a = 1.0f, "v_scale"_a = 1.0f,
          "stream"_a = nb::none());
    m.def("decode_attention", &mini_vllm_ext::decode_attention, "q"_a, "k"_a, "v"_a, "scale"_a, "causal"_a,
          "stream"_a = nb::none());
    m.def("flash_prefill", &mini_vllm_ext::flash_prefill, "q"_a, "k"_a, "v"_a, "scale"_a, "causal"_a,
          "stream"_a = nb::none());
    m.def("kv_quantize_scatter", &mini_vllm_ext::kv_quantize_scatter, "pool"_a, "slot_mapping"_a, "values"_a,
          "scale"_a, "stream"_a = nb::none());
}
