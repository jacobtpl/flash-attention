#include "utils.cuh"


#define A_IDX(i, j) ((i)*lda + (j))
#define B_IDX(i, j) ((i)*ldb + (j))
#define C_IDX(i, j) ((i)*ldc + (j))

__device__ inline void mma_16x8x8_tf32(float const *a, int lda, float const *b, int ldb, float *c, int ldc) {
    uint32_t t = threadIdx.x;
    uint32_t ai = t / 4;
    uint32_t aj = t % 4;

    uint32_t bj = t / 4;
    uint32_t bi = t % 4;
    
    uint32_t ci = (t*2) / 8;
    uint32_t cj = (t*2) % 8;

    uint32_t a0 = __float_as_uint(a[A_IDX(ai, aj)]);
    uint32_t a1 = __float_as_uint(a[A_IDX(ai + 8, aj)]);
    uint32_t a2 = __float_as_uint(a[A_IDX(ai, aj + 4)]);
    uint32_t a3 = __float_as_uint(a[A_IDX(ai + 8, aj + 4)]);
    
    uint32_t b0 = __float_as_uint(b[B_IDX(bi, bj)]);
    uint32_t b1 = __float_as_uint(b[B_IDX(bi + 4, bj)]);
    
    uint32_t c0 = __float_as_uint(c[C_IDX(ci, cj)]);
    uint32_t c1 = __float_as_uint(c[C_IDX(ci, cj) + 1]);
    uint32_t c2 = __float_as_uint(c[C_IDX(ci + 8, cj)]);
    uint32_t c3 = __float_as_uint(c[C_IDX(ci + 8, cj) + 1]);

    asm(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
            "{%0, %1, %2, %3},     /* 'D' matrix */\n"
            "{%4, %5, %6, %7},     /* 'A' matrix */\n"
            "{%8, %9},             /* 'B' matrix */\n"
            "{%10, %11, %12, %13}; /* 'C' matrix */"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );

    c[C_IDX(ci, cj)] = __uint_as_float(c0);
    c[C_IDX(ci, cj) + 1] = __uint_as_float(c1);
    c[C_IDX(ci + 8, cj)] = __uint_as_float(c2);
    c[C_IDX(ci + 8, cj) + 1] = __uint_as_float(c3);
}

__global__
void flash_attention_kernel_ptx_tf32(
    const float* Q,          // [B, H, N, D]
    const float* K,          // [B, H, N, D]
    const float* V,          // [B, H, N, D]
    const int N,            // sequence length
    const int d,            // hidden dimension
    const int num_col_tiles,           // number of column tiles
    const int num_row_tiles,           // number of row tiles
    const int col_tile_size,           // column tile size
    const int row_tile_size,           // row tile size
    const float scale,
    float* l,               // running sum [B, H, N]
    float* m,               // running max [B, H, N]
    float* O               // output [B, H, N, D]
) {
    const int WMMA_M = 16;
    const int WMMA_N = 8;
    const int WMMA_K = 8;

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
    float* tmp_result = (float*)&shmem[tile_size * 4];
    // printf("tile size: %d\n", tile_size);
    // printf("row tile size: %d, col tile size: %d\n", row_tile_size, col_tile_size);
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
            // Reset S to zero
            if (threadId < col_tile_size) {
                for (int y = 0; y < col_tile_size; y++) {
                    S[(col_tile_size * threadId) + y] = 0.0f;
                }
                for (int x = 0; x < d; x++) {
                    tmp_result[threadId * d + x] = 0.0f;
                }
            }
            __syncthreads();

            // Load Qi to shmem
            // Load l and m to registers
            if (threadId < row_tile_size) {
                for (int x = 0; x < d; x++) {
                    Qi[(threadId * d) + x] = Q[qkv_offset + (tile_size * i) + (threadId * d) + x];
                }
            }
            float row_m_prev = m[l_m_offset + (row_tile_size * i) + threadId];
            float row_l_prev = l[l_m_offset + (row_tile_size * i) + threadId];
            // Compute QK^T
            // Qi is 32 * 64, Kj is 32 * 64
            // if (threadId < col_tile_size) {
            //     for (int y = 0; y < col_tile_size; y++) {
            //         float sum = 0;
            //         for (int x = 0; x < d; x++) {
            //             // S[threadId, y] = Q[threadId, x] * KT[x, y]
            //             sum += Qi[(threadId * d) + x] * KjT[(x * col_tile_size) + y];
            //             // sum += Qi[(threadId * d) + x] * KjT[y * d + x];
            //         }
            //         S[(col_tile_size * threadId) + y] = sum;
            //     }
            // }
            // __syncthreads();

            for (int bi = 0; bi < col_tile_size; bi += WMMA_M) {
                for (int bj = 0; bj < row_tile_size; bj += WMMA_N) {
                    for (int k = 0; k < d; k += WMMA_K) {
                        mma_16x8x8_tf32(&Qi[bi * d + k], d, &KjT[k * col_tile_size + bj], col_tile_size, &S[bi * col_tile_size + bj], col_tile_size);
                    }
                }
            }
            // Print matrix S for debugging
            // if (threadId == 0 && blockIdx.x == 0 && blockIdx.y == 0 && j == 0 && i == 0) {
            //     printf("\nMatrix S:\n");
            //     for (int si = 0; si < row_tile_size; si++) {
            //         for (int sj = 0; sj < col_tile_size; sj++) {
            //             printf("%.4f ", S[si * col_tile_size + sj]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }
            // __syncthreads();
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
            for (int bi = 0; bi < col_tile_size; bi += WMMA_M) {
                for (int bj = 0; bj < d; bj += WMMA_N) {
                    for (int k = 0; k < col_tile_size; k += WMMA_K) {
                        mma_16x8x8_tf32(&S[bi * col_tile_size + k], col_tile_size, &Vj[k * d + bj], d, &tmp_result[bi * d + bj], d);
                    }
                }
            }

            if (threadId < col_tile_size) {
                // for (int x = 0; x < d; x++) {
                //     float pv = 0;
                //     for (int y = 0; y < col_tile_size; y++) {
                //         pv += S[(col_tile_size * threadId) + y] * Vj[y * d + x];
                //     }
                //     tmp_result[threadId * d + x] = pv;
                // }
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

void launch_flash_attention_ptx_tf32(
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
    const int shmem_size = (5 * col_tile_size * D * sizeof(float)) + (col_tile_size * row_tile_size * sizeof(float));
    cudaFuncSetAttribute(flash_attention_kernel_ptx_tf32, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024); // Set to 100KB

    dim3 grid(B, H); 
    dim3 block(col_tile_size); 
    flash_attention_kernel_ptx_tf32<<<grid, block, shmem_size>>>(
        Q, K, V, N, D, num_col_tiles, num_row_tiles, col_tile_size, row_tile_size, scale, d_l, d_m, O
    );
    
    cudaFree(d_l);
    cudaFree(d_m);
}