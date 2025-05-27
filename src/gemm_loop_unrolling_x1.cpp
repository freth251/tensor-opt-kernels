void gemm_loop_unrolling_x1(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; k+=2){
                sum += A[i * K + k] * B[k * N + j];
                sum += A[i * K + k+1] * B[(k+1) * N + j];
            }
                
            C[i * N + j] = sum;
        }
}
