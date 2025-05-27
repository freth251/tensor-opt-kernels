# Builiding a High Perforomance GEMM Kernel 

Accompanying write up : https://freth251.github.io/digital-garden/projects/tensor-opt-kernels/Building-a-High-Performance-GEMM-Kernel


Implementation of a GEMM kernel using various optimization techniques, such as `memory_aliasing`, `loop_unrolling`, `cache_blocking`, and `simd`. 


![GFLOPS vs Matrix size](gflops_cache_blocking_loop_unrolling_x1_loop_unrolling_x3_mem_aliasing_naive_simd.png)


![Execution time vs Matrix size](timesec_cache_blocking_loop_unrolling_x1_loop_unrolling_x3_mem_aliasing_naive_simd.png)
