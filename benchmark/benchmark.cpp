//
// Created by sayan on 9/20/25.
//

#include "einsum_single.hpp"
#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

auto gen_known_problem_and_solution() {
  std::vector vec{0, 1, 2, 3};
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  std::vector res_calc{130, 130, 130, 130, 230, 230, 230, 230,
                       330, 330, 330, 330, 430, 430, 430, 430};
}

template <std::size_t N, size_t M> auto generate_random_matrices() {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{1, 52};
  auto gen = [&]() { return dist(mersenne_engine); };

  std::vector<int> A(N * M);
  std::generate(A.begin(), A.end(), gen);
  std::vector<int> B(N * M);
  std::generate(B.begin(), B.end(), gen);

  std::mdspan<int, std::extents<size_t, N, M>> mdmat1{A.data()};
  std::mdspan<int, std::extents<size_t, N, M>> mdmat2{B.data()};
  return std::make_tuple(std::move(A), std::move(B), mdmat1, mdmat2);
}

#define MAKE_EINSUM_BENCH(Name, N, M)                                          \
  static void BM_Einsum##Name(benchmark::State &state) {                       \
    auto [A, B, mdA, mdB] = generate_random_matrices<N, M>();                  \
    make_einsum(ein, "ij,jk->ik", mdA, mdB);                                   \
    for (auto _ : state) {                                                     \
      ein.eval();                                                              \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_Einsum##Name)

#define MAKE_MATMUL_BENCH(Name, N, M)                                          \
  static void BM_MatMul##Name(benchmark::State &state) {                       \
    auto [A, B, mdA, mdB] = generate_random_matrices<N, M>();                  \
    std::vector<int> out(N * M);                                               \
    std::mdspan<int, std::extents<size_t, N, M>> mdC{out.data()};              \
    for (auto _ : state) {                                                     \
      for (auto i = 0; i < N; ++i) {                                           \
        for (auto k = 0; k < M; ++k) {                                         \
          mdC[i, k] = 0;                                                       \
          for (auto j = 0; j < N; ++j) {                                       \
            mdC[i, k] += mdA[i, j] * mdB[j, k];                                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_MatMul##Name)

/* Cache-optimal naive: i-j-k order keeps right[j,k] and output[i,k]
   sequential in the innermost k loop (row-major storage). */
#define MAKE_MATMUL_OPT_BENCH(Name, N, M)                                     \
  static void BM_MatMulOpt##Name(benchmark::State &state) {                   \
    auto [A, B, mdA, mdB] = generate_random_matrices<N, M>();                  \
    std::vector<int> out(N * M);                                               \
    std::mdspan<int, std::extents<size_t, N, M>> mdC{out.data()};              \
    for (auto _ : state) {                                                     \
      for (auto i = 0; i < N; ++i)                                             \
        for (auto k = 0; k < M; ++k)                                           \
          mdC[i, k] = 0;                                                       \
      for (auto i = 0; i < N; ++i)                                             \
        for (auto j = 0; j < N; ++j)                                           \
          for (auto k = 0; k < M; ++k)                                         \
            mdC[i, k] += mdA[i, j] * mdB[j, k];                                \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_MatMulOpt##Name)

MAKE_MATMUL_BENCH(2x2, 2, 2);
MAKE_MATMUL_BENCH(3x3, 3, 3);
MAKE_MATMUL_BENCH(4x4, 4, 4);
MAKE_MATMUL_BENCH(5x5, 5, 5);
MAKE_MATMUL_BENCH(6x6, 6, 6);
MAKE_MATMUL_BENCH(7x7, 7, 7);
MAKE_MATMUL_BENCH(8x8, 8, 8);
MAKE_MATMUL_BENCH(9x9, 9, 9);
MAKE_MATMUL_BENCH(10x10, 10, 10);
MAKE_MATMUL_BENCH(11x11, 11, 11);
MAKE_MATMUL_BENCH(12x12, 12, 12);

MAKE_MATMUL_OPT_BENCH(2x2, 2, 2);
MAKE_MATMUL_OPT_BENCH(3x3, 3, 3);
MAKE_MATMUL_OPT_BENCH(4x4, 4, 4);
MAKE_MATMUL_OPT_BENCH(5x5, 5, 5);
MAKE_MATMUL_OPT_BENCH(6x6, 6, 6);
MAKE_MATMUL_OPT_BENCH(7x7, 7, 7);
MAKE_MATMUL_OPT_BENCH(8x8, 8, 8);
MAKE_MATMUL_OPT_BENCH(9x9, 9, 9);
MAKE_MATMUL_OPT_BENCH(10x10, 10, 10);
MAKE_MATMUL_OPT_BENCH(11x11, 11, 11);
MAKE_MATMUL_OPT_BENCH(12x12, 12, 12);

MAKE_EINSUM_BENCH(2x2, 2, 2);
MAKE_EINSUM_BENCH(3x3, 3, 3);
MAKE_EINSUM_BENCH(4x4, 4, 4);
MAKE_EINSUM_BENCH(5x5, 5, 5);
MAKE_EINSUM_BENCH(6x6, 6, 6);
MAKE_EINSUM_BENCH(7x7, 7, 7);
MAKE_EINSUM_BENCH(8x8, 8, 8);
MAKE_EINSUM_BENCH(9x9, 9, 9);
MAKE_EINSUM_BENCH(10x10, 10, 10);
MAKE_EINSUM_BENCH(11x11, 11, 11);
MAKE_EINSUM_BENCH(12x12, 12, 12);

BENCHMARK_MAIN();