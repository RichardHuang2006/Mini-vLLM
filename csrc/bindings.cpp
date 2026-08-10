// The single pybind11 entry point for every kernel in csrc/.
//
// Each kernel step in Phase 3 and 4 adds one forward declaration and one m.def()
// here, so there is exactly one place that lists what the extension exposes.

#include <torch/extension.h>

torch::Tensor hello(const torch::Tensor& x, double a, double b);
torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps);
torch::Tensor rope(const torch::Tensor& x,
                   const torch::Tensor& positions,
                   const torch::Tensor& cos,
                   const torch::Tensor& sin);
torch::Tensor swiglu(const torch::Tensor& gate, const torch::Tensor& up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Mini-vLLM hand-written CUDA kernels";

  m.def("hello",
        &hello,
        "y = a*x + b, elementwise (Step 0.1 toolchain smoke kernel)",
        py::arg("x"),
        py::arg("a"),
        py::arg("b"));

  m.def("rmsnorm",
        &rmsnorm,
        "x * rsqrt(mean(x^2) + eps) * weight over the last dimension (Step 3.1)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("eps") = 1e-6);

  m.def("rope",
        &rope,
        "rotary position embedding at explicit positions, tables gathered in-kernel (Step 3.2)",
        py::arg("x"),
        py::arg("positions"),
        py::arg("cos"),
        py::arg("sin"));

  m.def("swiglu",
        &swiglu,
        "silu(gate) * up, the elementwise half of the MLP (Step 3.3)",
        py::arg("gate"),
        py::arg("up"));
}
