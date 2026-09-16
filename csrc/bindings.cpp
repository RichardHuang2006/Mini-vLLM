// The single pybind11 entry point for every kernel in csrc/.

#include <torch/extension.h>

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps);
torch::Tensor rope(const torch::Tensor& x,
                   const torch::Tensor& positions,
                   const torch::Tensor& cos,
                   const torch::Tensor& sin);
torch::Tensor swiglu(const torch::Tensor& gate, const torch::Tensor& up);
torch::Tensor decode_attention(const torch::Tensor& q,
                               const torch::Tensor& k,
                               const torch::Tensor& v,
                               double scale);
torch::Tensor flash_prefill(const torch::Tensor& q,
                            const torch::Tensor& k,
                            const torch::Tensor& v,
                            double scale);
torch::Tensor paged_attention(const torch::Tensor& q,
                              const torch::Tensor& key_pool,
                              const torch::Tensor& value_pool,
                              const torch::Tensor& block_tables,
                              const torch::Tensor& cu_seqlens_q,
                              const torch::Tensor& context_lens,
                              const torch::Tensor& seq_lens,
                              int64_t max_query_len,
                              int64_t max_context_len,
                              double scale,
                              double k_scale,
                              double v_scale);
void kv_quantize_scatter(const torch::Tensor& key,
                         const torch::Tensor& value,
                         torch::Tensor key_pool,
                         torch::Tensor value_pool,
                         const torch::Tensor& slot_mapping,
                         double k_scale,
                         double v_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Mini-vLLM hand-written CUDA kernels";

  m.def("rmsnorm",
        &rmsnorm,
        "x * rsqrt(mean(x^2) + eps) * weight over the last dimension",
        py::arg("x"),
        py::arg("weight"),
        py::arg("eps") = 1e-6);

  m.def("rope",
        &rope,
        "rotary position embedding at explicit positions, tables gathered in-kernel",
        py::arg("x"),
        py::arg("positions"),
        py::arg("cos"),
        py::arg("sin"));

  m.def("swiglu",
        &swiglu,
        "silu(gate) * up, the elementwise half of the MLP",
        py::arg("gate"),
        py::arg("up"));

  m.def("decode_attention",
        &decode_attention,
        "grouped attention for a single query token, via online softmax",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scale"));

  m.def("flash_prefill",
        &flash_prefill,
        "tiled causal grouped attention for many query tokens",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scale"));

  m.def("paged_attention",
        &paged_attention,
        "ragged decode + prefill attention over a paged KV cache, gathering "
        "through the block table inside the kernel",
        py::arg("q"),
        py::arg("key_pool"),
        py::arg("value_pool"),
        py::arg("block_tables"),
        py::arg("cu_seqlens_q"),
        py::arg("context_lens"),
        py::arg("seq_lens"),
        py::arg("max_query_len"),
        py::arg("max_context_len"),
        py::arg("scale"),
        py::arg("k_scale") = 1.0,
        py::arg("v_scale") = 1.0);

  m.def("kv_quantize_scatter",
        &kv_quantize_scatter,
        "fused quantize-and-scatter of a step's keys and values into an FP8 paged cache",
        py::arg("key"),
        py::arg("value"),
        py::arg("key_pool"),
        py::arg("value_pool"),
        py::arg("slot_mapping"),
        py::arg("k_scale"),
        py::arg("v_scale"));
}
