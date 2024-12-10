#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

// Declaration of CUDA kernel launcher
void launch_flash_attention(const float* Q, const float* K, const float* V, float* O,
                          const int B, const int H, const int N, const int D);

// PyTorch binding
torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    
    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);
    
    auto options = torch::TensorOptions()
        .dtype(q.dtype())
        .device(q.device());
    auto output = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    
    launch_flash_attention(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention forward");
}
