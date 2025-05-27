#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <map>
#include <string>

using GemmFunc = void (*)(const float*, const float*, float*, int, int, int);

extern void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_mem_aliasing(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_loop_unrolling_x1(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_loop_unrolling_x3(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_cache_blocking(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_simd(const float* A, const float* B, float* C, int M, int N, int K);

struct BenchmarkResult {
    double gflops;
    double time_sec;
};

std::map<std::string, GemmFunc> gemm_funcs = {
    {"naive", gemm_naive},
    {"mem_aliasing", gemm_mem_aliasing},
    {"loop_unrolling_x1", gemm_loop_unrolling_x1},
    {"loop_unrolling_x3", gemm_loop_unrolling_x3}, 
    {"cache_blocking", gemm_cache_blocking}, 
    {"simd", gemm_simd}
};

BenchmarkResult run_benchmark_avg(GemmFunc gemm, int M, int N, int K, int runs = 5) {
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    double total_time = 0.0;
    for (int i = 0; i < runs; ++i) {
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();
        gemm(A.data(), B.data(), C.data(), M, N, K);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    double avg_time = total_time / runs;
    double avg_gflops = 2.0 * M * N * K / (avg_time * 1e9);

    return {avg_gflops, avg_time};
}
int main() {
    std::vector<int> sizes = {128, 256, 512, 768, 1024};

    std::ofstream fout("benchmark_results.csv");
    fout << "Function,M,N,K,GFLOPS,TimeSec\n";


    for (const auto& [name, func] : gemm_funcs) {
        for (int sz : sizes) {
            BenchmarkResult result = run_benchmark_avg(func, sz, sz, sz, 5);
            std::cout << name << ": " << sz << "x" << sz
                      << ", GFLOPS = " << result.gflops
                      << ", Time = " << result.time_sec << "s\n";
            fout << name << "," << sz << "," << sz << "," << sz
                 << "," << result.gflops << "," << result.time_sec << "\n";
        }
    }

    fout.close();
    return 0;
}
