CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17

all: main

main: src/main.cpp src/gemm_naive.cpp src/gemm_mem_aliasing.cpp src/gemm_loop_unrolling_x1.cpp src/gemm_loop_unrolling_x3.cpp src/gemm_cache_blocking.cpp
	$(CXX) $(CXXFLAGS) -I./src $^ -o main

clean:
	rm -f main
