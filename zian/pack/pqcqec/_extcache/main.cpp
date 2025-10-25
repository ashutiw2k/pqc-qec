#include <torch/extension.h>

#include <torch/extension.h>
torch::Tensor fused_base_noise_segment(
    torch::Tensor states, torch::Tensor scratch,
    torch::Tensor gate_kind, torch::Tensor q1s, torch::Tensor q2s,
    torch::Tensor rz1, torch::Tensor rx1, torch::Tensor rz2, torch::Tensor rx2,
    int reverse);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("fused_base_noise_segment", torch::wrap_pybind_function(fused_base_noise_segment), "fused_base_noise_segment");
}