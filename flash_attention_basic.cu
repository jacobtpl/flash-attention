#include "utils.cuh"

__global__
void flash_attention_kernel_basic(
    const float* Q,   // [B, H, N, D]
    const float* K,     // [B, H, N, D]
    const float* V,      // [B, H, N, D]
    const int N,       // sequence length
    const int d,   // hidden dimension
    const int num_col_tiles,    
    const int num_row_tiles,   
    const int col_tile_size,   
    const int row_tile_size,   
    const float scale,
    float* l,      // running sum [B, H, N]
    float* m,      // running max [B, H, N]
    float* O       // output [B, H, N, D]
) {
    int threadId = threadIdx.x;
    int batch_idx = blockIdx.x; int head_idx = blockIdx.y;

    int qkv_offset = (batch_idx * gridDim.y * N * d) + (head_idx * N * d);
    int l_m_offset = (batch_idx * gridDim.y * N) + (head_idx * N);

    extern __shared__ float shmem[];
    int tile_size = col_tile_size * d;
    float* Qi = shmem;
    float* Kj = &shmem[tile_size];
    float* Vj = &shmem[tile_size * 2];
    float* S = &shmem[tile_size * 3];

    for (int j = 0; j < num_col_tiles; j++) {
        // Load in Kj, Vj to shmem
        for (int x = 0; x < d; x++) {
            int shmem_idx = (threadId * d) + x;
            int idx = qkv_offset + (tile_size * j) + shmem_idx;
            Kj[shmem_idx] = K[idx];
            Vj[shmem_idx] = V[idx];
        }
        __syncthreads();

        for (int i = 0; i < num_row_tiles; i++) {
            // Load Qi to shmem
            // Load l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(threadId * d) + x] = Q[qkv_offset + (tile_size * i) + (threadId * d) + x];
            }
            float row_m_prev = m[l_m_offset + (row_tile_size * i) + threadId];
            float row_l_prev = l[l_m_offset + (row_tile_size * i) + threadId];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < col_tile_size; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(threadId * d) + x] * Kj[(y * d) + x];
                }
                sum *= scale;
                S[(col_tile_size * threadId) + y] = sum;
                row_m = max(row_m, sum);
            }

            float row_l = 0;
            for (int y = 0; y < col_tile_size; y++) {
                S[(col_tile_size * threadId) + y] = __expf(S[(col_tile_size * threadId) + y] - row_m);
                row_l += S[(col_tile_size * threadId) + y];
            }

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + 
                             (__expf(row_m - row_m_new) * row_l);

            // Update O, l, m
            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < col_tile_size; y++) {
                    pv += S[(col_tile_size * threadId) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (threadId * d) + x] = 
                    (1 / row_l_new) * (
                        (row_l_prev * __expf(row_m_prev - row_m_new) * 
                         O[qkv_offset + (tile_size * i) + (threadId * d) + x]) +
                        (__expf(row_m - row_m_new) * pv)
                    );
            }
            m[l_m_offset + (row_tile_size * i) + threadId] = row_m_new;
            l[l_m_offset + (row_tile_size * i) + threadId] = row_l_new;
        }
        __syncthreads();
    }
}

void launch_flash_attention_basic(
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
    const int shmem_size = (3 * col_tile_size * D * sizeof(float)) + (col_tile_size * row_tile_size * sizeof(float));
    int max_shmem_size;
    cudaDeviceGetAttribute(&max_shmem_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    // printf("Max shared memory: %d, requested shared memory: %d\n", max_shmem_size, shmem_size);
    
    dim3 grid(B, H); 
    dim3 block(col_tile_size); 
    
    flash_attention_kernel_basic<<<grid, block, shmem_size>>>(
        Q, K, V, N, D, num_col_tiles, num_row_tiles, col_tile_size, row_tile_size, scale, d_l, d_m, O
    );
    
    cudaFree(d_l);
    cudaFree(d_m);
}