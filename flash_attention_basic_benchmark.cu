#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <algorithm>  // for min_element, max_element
#include <numeric>    // for accumulate
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "flash_attention_basic.cu"
#include "utils.cuh"

void flash_attention_cpu(const float* Q, const float* K, const float* V, float* output, int B, int N, int H, int d) {
    std::vector<float> scores(N * N);
    bool debug = false;

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q = Q + b * H * N * d + h * N * d;
            const float* k = K + b * H * N * d + h * N * d;
            const float* v = V + b * H * N * d + h * N * d;
            float* o = output + b * H * N * d + h * N * d;

            // Compute Q * K^T (row-major)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int d_i = 0; d_i < d; ++d_i) {
                        // Q[i,d_i] * K[j,d_i] for row-major layout
                        sum += q[i * d + d_i] * k[j * d + d_i];
                    }
                    scores[i * N + j] = sum / sqrtf(d);
                    
                }
            }

            if (debug) {
                //print scores
                std::cout << "CPU Scores: ";
                for (int i = 0; i < N * N; ++i) {
                    std::cout << scores[i] << " ";
                }
                std::cout << std::endl;
            }

            // Apply softmax row by row
            for (int i = 0; i < N; ++i) {
                // first find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < N; ++j) {
                    max_val = std::max(max_val, scores[i * N + j]);
                }

                // compute exp and sum
                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    scores[i * N + j] = expf(scores[i * N + j] - max_val);
                    sum_exp += scores[i * N + j];
                }

                // normalize
                for (int j = 0; j < N; ++j) {
                    scores[i * N + j] /= sum_exp;
                }
            }

            if (debug) {
                // print cpu softmax
                std::cout << "CPU Softmax: ";
                for (int i = 0; i < N * N; ++i) {
                    std::cout << scores[i] << " ";
                }
                std::cout << std::endl;
            }

            // Compute scores * V (row-major)
            for (int i = 0; i < N; ++i) {
                for (int d_i = 0; d_i < d; ++d_i) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        // scores[i,j] * V[j,d_i] for row-major layout
                        sum += scores[i * N + j] * v[j * d + d_i];
                    }
                    o[i * d + d_i] = sum;
                }
            }
        }
    }
}

void initialize_tensor(std::vector<float>& tensor, int size) {
    for (int i = 0; i < size; ++i) {
        tensor[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

struct BenchmarkResults {
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    std::vector<double> individual_times;
};

BenchmarkResults run_cpu_benchmark(const std::vector<float>& h_Q, 
                                 const std::vector<float>& h_K,
                                 const std::vector<float>& h_V,
                                 std::vector<float>& h_output,
                                 int B, int N, int H, int d,
                                 int num_iterations) {
    BenchmarkResults results;
    results.individual_times.reserve(num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        flash_attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_output.data(), B, N, H, d);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        results.individual_times.push_back(time_ms);
    }

    // Calculate statistics
    results.min_time_ms = *std::min_element(results.individual_times.begin(), results.individual_times.end());
    results.max_time_ms = *std::max_element(results.individual_times.begin(), results.individual_times.end());
    results.avg_time_ms = std::accumulate(results.individual_times.begin(), results.individual_times.end(), 0.0) / num_iterations;

    return results;
}

__global__ void columnwise_softmax(float* input, float* output, int N) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one column
    if (tid < N) {
        // Find max element in the column for numerical stability
        float max_val = input[tid];
        for (int i = 1; i < N; i++) {
            max_val = fmaxf(max_val, input[i * N + tid]);
        }
        
        // Calculate exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float val = expf(input[i * N + tid] - max_val);
            output[i * N + tid] = val;
            sum += val;
        }
        
        // Normalize by sum
        for (int i = 0; i < N; i++) {
            output[i * N + tid] /= sum;
        }
    }
}

// V is specifically in column-major format
BenchmarkResults run_gpu_benchmark(cublasHandle_t handle,
                                 float* d_Q, float* d_K, float* d_V, float* d_output,
                                 int B, int N, int H, int d,
                                 int num_iterations) {
    BenchmarkResults results;
    bool debug = false;
    results.individual_times.reserve(num_iterations);

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Calculate leading dimensions and strides
    int ldq = d;      
    int ldk = d;      
    int lds = N;    

    for (int i = 0; i < num_iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        float alpha_qk = 1.0f / sqrtf(d);
        float alpha_v = 1.0f;
        float beta = 0.0f;

        float* d_scores;
        CUDA_CHECK(cudaMalloc(&d_scores, B * H * N * N * sizeof(float)));

        // Process each batch and head
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                float* current_Q = d_Q + (b * H * N * d) + (h * N * d);
                float* current_K = d_K + (b * H * N * d) + (h * N * d);
                float* current_V = d_V + (b * H * N * d) + (h * N * d);
                float* current_scores = d_scores + (b * H * N * N) + (h * N * N);
                float* current_output = d_output + (b * H * N * d) + (h * N * d);

                // Q * K^T computation
                CUBLAS_CHECK(cublasSgemm(handle,
                    CUBLAS_OP_T,  
                    CUBLAS_OP_N, 
                    N, N, d,        // m, n, k dimensions
                    &alpha_qk,      // scaling factor
                    current_Q, ldq,  // Q matrix, N x d
                    current_K, ldk,  // K matrix, N x d
                    &beta,   
                    current_scores, lds)); // output scores which is N x N

                if (debug) {
                    // print current scores
                    float* h_scores = new float[N * N];
                    CUDA_CHECK(cudaMemcpy(h_scores, current_scores, N * N * sizeof(float), cudaMemcpyDeviceToHost));
                    std::cout << "Scores: ";
                    for (int i = 0; i < N * N; ++i) {
                        std::cout << h_scores[i] << " ";
                    }
                    std::cout << std::endl;
                }

                int threadsPerBlock = std::min(N, 1024);
                int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
                columnwise_softmax<<<blocksPerGrid, threadsPerBlock>>>(current_scores, current_scores, N);
                CUDA_CHECK(cudaGetLastError());

                if (debug) {
                    // print gpu softmax
                    float* h_scores_softmax = new float[N * N];
                    CUDA_CHECK(cudaMemcpy(h_scores_softmax, current_scores, N * N * sizeof(float), cudaMemcpyDeviceToHost));
                    std::cout << "GPU Softmax: ";
                    for (int i = 0; i < N * N; ++i) {
                        std::cout << h_scores_softmax[i] << " ";
                    }
                    std::cout << std::endl;
                }
                
                CUBLAS_CHECK(cublasSgemm(handle,
                    CUBLAS_OP_N,  
                    CUBLAS_OP_N, 
                    N, d, N,        // m, n, k dimensions
                    &alpha_v,       // same scale
                    current_scores, lds, // scores matrix which are N x N
                    current_V, N,     // V matrix is N x d
                    &beta,    
                    current_output, N)); // output matrix is N x d
                
            }
        }

        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        results.individual_times.push_back(time_ms);
    }

    std::sort(results.individual_times.begin(), results.individual_times.end());
    results.min_time_ms = results.individual_times.front();
    results.max_time_ms = results.individual_times.back();
    results.avg_time_ms = std::accumulate(results.individual_times.begin(), results.individual_times.end(), 0.0) / num_iterations;
    //results.median_time_ms = results.individual_times[num_iterations / 2];

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return results;
}

BenchmarkResults run_flash_attention_benchmark(
    float* d_Q, float* d_K, float* d_V, float* d_output,
    int B, int N, int H, int d,
    int num_iterations,
    void (*launch_flash_attention_fn)(const float*, const float*, const float*, float*, const int, const int, const int, const int)) {
    
    BenchmarkResults results;
    results.individual_times.reserve(num_iterations);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < num_iterations; ++i) {
        // Zero out output buffer before each iteration
        CUDA_CHECK(cudaMemset(d_output, 0, B * H * N * d * sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        launch_flash_attention_fn(d_Q, d_K, d_V, d_output, B, H, N, d);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        results.individual_times.push_back(time_ms);
    }

    // Calculate statistics
    results.min_time_ms = *std::min_element(results.individual_times.begin(), results.individual_times.end());
    results.max_time_ms = *std::max_element(results.individual_times.begin(), results.individual_times.end());
    results.avg_time_ms = std::accumulate(results.individual_times.begin(), results.individual_times.end(), 0.0) / num_iterations;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return results;
}

void print_benchmark_results(const std::string& name, const BenchmarkResults& results) {
    std::cout << "\n=== " << name << " Performance ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << results.avg_time_ms << " ms" << std::endl;
    std::cout << "Min time:     " << results.min_time_ms << " ms" << std::endl;
    std::cout << "Max time:     " << results.max_time_ms << " ms" << std::endl;
    std::cout << "Variance:     " << std::fixed << std::setprecision(6)
              << std::accumulate(results.individual_times.begin(), results.individual_times.end(), 0.0,
                               [&](double acc, double x) {
                                   return acc + (x - results.avg_time_ms) * (x - results.avg_time_ms);
                               }) / results.individual_times.size() << std::endl;
}

void initialize_cuda_device() {
    // Initialize CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(-1);
    }

    bool print_available_devices = false;
    if (print_available_devices) {
        // Print available devices
        std::cout << "Available CUDA devices:" << std::endl;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            std::cout << "Device " << i << ": " << prop.name 
                    << " (Compute " << prop.major << "." << prop.minor << ")" 
                    << "\n\tGlobal Memory: " << prop.totalGlobalMem / (1024*1024) << " MB"
                    << "\n\tSMs: " << prop.multiProcessorCount
                    << "\n\tMax threads per block: " << prop.maxThreadsPerBlock
                    << "\n\tMax threads per SM: " << prop.maxThreadsPerMultiProcessor
                    << std::endl;
        }
    }

    // Select first device by default
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nUsing device 0: " << prop.name << std::endl;
}

// Returns RRMSE
float compare_results(std::vector<float>& h1, std::vector<float>& h2, int32_t output_size) {
    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < output_size; ++i) {
        float diff = h2[i] - h1[i];
        mse += diff * diff;
        ref_mean_square += h1[i] * h1[i];
    }
    mse /= output_size;
    ref_mean_square /= output_size;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);
    return rel_rmse;
}

int main() {
    srand(0);
    initialize_cuda_device();
    bool run_cpu = true;

    const int B = 32;    // Batch size
    const int N = 64;   // Sequence length
    const int H = 8;    // Number of attention heads
    const int d = 64;   // Dimension per head
    const int num_iterations = 10;  // Number of benchmark iterations

    const int QKV_size = B * N * H * d;
    const int output_size = QKV_size;

    // Host tensors
    std::vector<float> h_Q(QKV_size), h_K(QKV_size), h_V(QKV_size), h_V_col(QKV_size);
    std::vector<float> h_output_cpu(output_size), h_output_gpu(output_size), h_output_gpu_base(output_size), h_output_flash(output_size);

    initialize_tensor(h_Q, QKV_size);
    initialize_tensor(h_K, QKV_size);
    initialize_tensor(h_V, QKV_size);

    // Set V column major
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < d; ++j) {
                    h_V_col[b * H * N * d + h * N * d + j * N + i] = h_V[b * H * N * d + h * N * d + i * d + j];
                }
            }
        }
    }

    // Device tensors
    float *d_Q, *d_K, *d_V, *d_output, *d_V_col;
    CUDA_CHECK(cudaMalloc(&d_Q, QKV_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, QKV_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, QKV_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_col, QKV_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), QKV_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), QKV_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), QKV_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_col, h_V_col.data(), QKV_size * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Warmup runs
    flash_attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_output_cpu.data(), B, N, H, d);
    cudaDeviceSynchronize();

    // Run benchmarks
    std::cout << "\nRunning benchmarks with " << num_iterations << " iterations..." << std::endl;
    std::cout << "Configuration: B=" << B << ", N=" << N << ", H=" << H << ", d=" << d << std::endl;

    BenchmarkResults cpu_results;
    if (run_cpu) {
        cpu_results = run_cpu_benchmark(h_Q, h_K, h_V, h_output_cpu, B, N, H, d, num_iterations);
    }

    BenchmarkResults gpu_results;
    gpu_results = run_gpu_benchmark(handle, d_Q, d_K, d_V_col, d_output, B, N, H, d, num_iterations);
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Column major to row major
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < d; ++j) {
                    h_output_gpu_base[b * H * N * d + h * N * d + i * d + j] = h_output_gpu[b * H * N * d + h * N * d + j * N + i];
                }
            }
        }
    }

    auto flash_results = run_flash_attention_benchmark(d_Q, d_K, d_V, d_output, B, N, H, d, num_iterations, launch_flash_attention_basic);
    CUDA_CHECK(cudaMemcpy(h_output_flash.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    if (run_cpu) {
        print_benchmark_results("CPU", cpu_results);
    }
    print_benchmark_results("GPU", gpu_results);
    print_benchmark_results("Flash", flash_results);
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    if (run_cpu) {
        float rrmse_cpu_gpubase = compare_results(h_output_cpu, h_output_gpu_base, output_size);
        float rrmse_cpu_flash = compare_results(h_output_cpu, h_output_flash, output_size);
        std::cout << "RRMSE CPU vs GPU Base: " << rrmse_cpu_gpubase << std::endl;
        std::cout << "RRMSE CPU vs Flash:    " << rrmse_cpu_flash << std::endl;
    } else {
        float rrmse_gpu_flash = compare_results(h_output_gpu_base, h_output_flash, output_size);
        std::cout << "RRMSE GPU Base vs Flash: " << rrmse_gpu_flash << std::endl;
    }

    if (run_cpu) {
        double speedup = cpu_results.min_time_ms / gpu_results.min_time_ms;
        std::cout << "\nGPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    double flash_speedup = gpu_results.min_time_ms / flash_results.min_time_ms;
    std::cout << "Flash Speedup: " << std::fixed << std::setprecision(2) << flash_speedup << "x" << std::endl;

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_V_col));

    return 0;
}