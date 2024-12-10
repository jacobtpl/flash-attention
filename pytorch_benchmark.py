import math
from typing import List, Tuple

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def load_cuda_module():
    return load(
        name="flash_attention_ptx_fp8",
        sources=["pytorch_flash_attention_main.cpp", "flash_attention_ptx_fp8.cu"], # May need to change these depending on naming
        extra_cuda_cflags=["-O2", "--gpu-architecture=sm_90", "--ptxas-options=-v"],
        verbose=True,
        build_directory="build",
    )

def manual_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def benchmark_attention(
    batch_size: int,
    n_head: int,
    seq_len: int,
    head_embd: int,
) -> Tuple[float, float, bool]:
    # Initialize tensors
    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    
    # Profile manual attention
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        manual_result = manual_attn(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

    # Profile flash attention
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        minimal_result = minimal_attn.forward(q, k, v)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    
    # Check correctness
    correctness = torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02)
    return

# Actually we are only testing one size for now because outputs are very long
def test_multiple_sizes():
    global minimal_attn
    minimal_attn = load_cuda_module()
    
    configs = [
        # batch_size, n_head, seq_len, head_embd
        (1024, 12, 64, 64),
    ]
    
    print("\nBenchmarking different problem sizes:")
    print("-" * 80)
    print(f"{'Config':40} | {'Manual (ms)':12} | {'Flash (ms)':12} | {'Speedup':8} | {'Correct'}")
    print("-" * 80)
    
    for batch_size, n_head, seq_len, head_embd in configs:
        benchmark_attention(
            batch_size, n_head, seq_len, head_embd
        )


if __name__ == "__main__":
    test_multiple_sizes()
