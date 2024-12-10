#include "utils.cuh"
#include <mma.h>

using namespace nvcuda::wmma;

__global__
void flash_attention_kernel_wmma_fp16(
    const float* Q,              // [B, H, N, D]
    const float* K,              // [B, H, N, D] 
    const float* V,              // [B, H, N, D]
    const int N,                 // sequence length
    const int d,                 // hidden dimension
    const int num_col_tiles,     // number of column tiles
    const int num_row_tiles,     // number of row tiles
    const int col_tile_size,     // column tile size
    const int row_tile_size,     // row tile size
    const float scale,           // 1 / sqrt(d)
    float* l,                    // running sum [B, H, N]
    float* m,                    // running max [B, H, N]
    float* O                     // output [B, H, N, D]
) {
    int threadId = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int l_m_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    extern __shared__ float shmem[];
    int tile_size = col_tile_size * d;
    float* Qi = shmem;
    float* KjT = &shmem[tile_size];
    float* Vj = &shmem[tile_size * 2];
    float* S = &shmem[tile_size * 3];
    half* Qi_half = (half*)&shmem[tile_size * 4];
    half* KjT_half = (half*)&shmem[tile_size * 5];
    half* S_half = (half*)&shmem[tile_size * 6];
    half* Vj_half = (half*)&shmem[tile_size * 7];
    float* tmp_result = (float*)&shmem[tile_size * 8];
    
    for (int j = 0; j < num_col_tiles; j++) {
        // Load in Kj, Vj to shmem
        if (threadId < col_tile_size) {
            for (int x = 0; x < d; x++) {
                // int shmem_idx = (threadId * d) + x;
                int shmem_idx = x * col_tile_size + threadId;
                int idx = qkv_offset + (j * col_tile_size + threadId) * d + x;  // Row-major input
                KjT[shmem_idx] = K[idx];
            }
            for (int x = 0; x < d; x++) {
                int shmem_idx = threadId * d + x;
                int idx = qkv_offset + (j * col_tile_size + threadId) * d + x;  // Row-major input
                Vj[shmem_idx] = V[idx];
            }
        }
        __syncthreads();

        for (int i = 0; i < num_row_tiles; i++) {
            // Load Qi to shmem
            // Load l and m to registers
            if (threadId < row_tile_size) {
                for (int x = 0; x < d; x++) {
                    Qi[(threadId * d) + x] = Q[qkv_offset + (tile_size * i) + (threadId * d) + x];
                }
            }
            float row_m_prev = m[l_m_offset + (row_tile_size * i) + threadId];
            float row_l_prev = l[l_m_offset + (row_tile_size * i) + threadId];

            // Configure WMMA fragment sizes
            const int WMMA_M = 16;
            const int WMMA_N = 16;
            const int WMMA_K = 16;  // For FP16

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

            if (threadId < col_tile_size) {
                for (int x = 0; x < d; x++) {
                    Qi_half[threadId * d + x] = __float2half(Qi[threadId * d + x]);
                    KjT_half[x * col_tile_size + threadId] = __float2half(KjT[x * col_tile_size + threadId]);
                }
            }
            __syncthreads();

            for (int bi = 0; bi < col_tile_size; bi += WMMA_M) {
                for (int bj = 0; bj < row_tile_size; bj += WMMA_N) {
                    fill_fragment(c_frag, 0.0f);
                    for (int k = 0; k < d; k += WMMA_K) {
                        if (k + WMMA_K <= d) {
                            load_matrix_sync(a_frag, &Qi_half[bi * d + k], d);
                            load_matrix_sync(b_frag, &KjT_half[k * col_tile_size + bj], col_tile_size);
                            mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    store_matrix_sync(&S[bi * col_tile_size + bj], c_frag, col_tile_size, mem_row_major);
                }
            }

            float row_m = -INFINITY;
            float row_l = 0;
            float row_m_new, row_l_new;

            if (threadId < col_tile_size) {
                // Scale S
                for (int y = 0; y < col_tile_size; y++) {
                    S[(col_tile_size * threadId) + y] *= scale;
                }

                // Then compute row max
                for (int y = 0; y < col_tile_size; y++) {
                    row_m = max(row_m, S[(col_tile_size * threadId) + y]);
                }

                for (int y = 0; y < col_tile_size; y++) {
                    S[(col_tile_size * threadId) + y] = __expf(S[(col_tile_size * threadId) + y] - row_m);
                    row_l += S[(col_tile_size * threadId) + y];
                }

                row_m_new = max(row_m_prev, row_m);
                row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + 
                            (__expf(row_m - row_m_new) * row_l);
            }
            __syncthreads();
            // Update O, l, m
            
            // S: N x N, Vj: N x d
            // tmp_result: N x d

            if (threadId < col_tile_size) {
                // Convert S matrix (maintaining row-major layout)
                for (int y = 0; y < col_tile_size; y++) {
                    S_half[threadId * col_tile_size + y] = __float2half(S[threadId * col_tile_size + y]);
                }
                // Convert V matrix (maintaining row-major layout)
                for (int x = 0; x < d; x++) {
                    Vj_half[threadId * d + x] = __float2half(Vj[threadId * d + x]);
                }
            }
            __syncthreads();

            for (int bi = 0; bi < col_tile_size; bi += WMMA_M) {
                for (int bj = 0; bj < d; bj += WMMA_N) {
                    fill_fragment(c_frag, 0.0f);
                    for (int k = 0; k < col_tile_size; k += WMMA_K) {
                        if (k + WMMA_K <= col_tile_size) {
                            load_matrix_sync(a_frag, &S_half[bi * col_tile_size + k], col_tile_size);
                            load_matrix_sync(b_frag, &Vj_half[k * d + bj], d);
                            mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                    }
                    store_matrix_sync(&tmp_result[bi * d + bj], c_frag, d, mem_row_major);
                }
            }

            if (threadId < col_tile_size) {
                for (int x = 0; x < d; x++) {
                    O[qkv_offset + (tile_size * i) + (threadId * d) + x] = 
                        (1 / row_l_new) * (
                            (row_l_prev * __expf(row_m_prev - row_m_new) * 
                            O[qkv_offset + (tile_size * i) + (threadId * d) + x]) +
                            (__expf(row_m - row_m_new) * tmp_result[threadId * d + x])
                        );
                }
                m[l_m_offset + (row_tile_size * i) + threadId] = row_m_new;
                l[l_m_offset + (row_tile_size * i) + threadId] = row_l_new;
            }
        }
        __syncthreads();
    }
}

void launch_flash_attention_wmma_fp16(
    const float* Q, const float* K, const float* V, float* O,
    const int B, const int H, const int N, const int D
) {
    const int col_tile_size = 32;
    const int row_tile_size = 32;
    
    const int num_col_tiles = (N + col_tile_size - 1) / col_tile_size; 
    const int num_row_tiles = (N + row_tile_size - 1) / row_tile_size;
    const float scale = 1.0f / sqrt(D);
    
    // Allocate and initialize running statistics
    float *d_l, *d_m;
    CUDA_CHECK(cudaMalloc(&d_l, B * H * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m, B * H * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_l, 0, B * H * N * sizeof(float)));
    float neg_inf = -INFINITY;
    CUDA_CHECK(cudaMemset(d_m, neg_inf, B * H * N * sizeof(float)));
    
    // Calculate shared memory size
    const int shmem_size = (10 * col_tile_size * D * sizeof(float)) + (col_tile_size * row_tile_size * sizeof(float));
    cudaFuncSetAttribute(flash_attention_kernel_wmma_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024); // Set to 100KB

    dim3 grid(B, H); 
    dim3 block(col_tile_size); 
    flash_attention_kernel_wmma_fp16<<<grid, block, shmem_size>>>(
        Q, K, V, N, D, num_col_tiles, num_row_tiles, col_tile_size, row_tile_size, scale, d_l, d_m, O
    );
    
    cudaFree(d_l);
    cudaFree(d_m);
}