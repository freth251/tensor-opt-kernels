#include <algorithm>
#include <xsimd/xsimd.hpp>
#include <vector>

using namespace xsimd;


constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;

void gemm_simd(const float* A, const float* B, float* C, int M, int N, int K) {
    using batch_type = xsimd::batch<float>;
    constexpr std::size_t simd_size = batch_type::size;


    for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {

                int i_max = std::min(i0 + BLOCK_M, M);
                int j_max = std::min(j0 + BLOCK_N, N);
                int k_max = std::min(k0 + BLOCK_K, K);

                for (int i = i0; i < i_max; ++i) {
                    for (int j = j0; j < j_max; j += simd_size) {
                        batch_type c_vec = batch_type::load_unaligned(&C[i * N + j]);
                        for (int k = k0; k < k_max; k++) {
                            batch_type b_vec = batch_type::load_unaligned(&B[k * N + j]);
                            float a_scalar = A[i * K + k];
                            c_vec = xsimd::fma(batch_type(a_scalar), b_vec, c_vec);

                        }
                        c_vec.store_unaligned(&C[i * N + j]);

                    }
                    // Fallback for leftover columns
                    for (int j = j_max - (j_max % simd_size); j < j_max; ++j) {
                        float sum = C[i * N + j];
                        for (int k = 0; k < K; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}
