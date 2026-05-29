//
// Created by sayan on 9/20/25.
//

#include "einsum.hpp"
#include <benchmark/benchmark.h>
#include <experimental/mdspan>

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

template <std::size_t N, size_t M> constexpr auto generate_random_matrices() {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::uniform_int_distribution<int> dist{1, 52};
  auto gen = [&]() { return dist(mersenne_engine); };
  std::array<int, N * M> A{};
  std::generate(A.begin(), A.end(), gen);

  std::vector<int> B(N * M);
  std::generate(B.begin(), B.end(), gen);

  std::mdspan<int, std::extents<size_t, N, M>> mdmat1{A.data()};
  std::mdspan<int, std::extents<size_t, N, M>> mdmat2{B.data()};
  return std::make_tuple(A, B, mdmat1, mdmat2);
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

MAKE_MATMUL_BENCH(A, 2, 2);
MAKE_MATMUL_BENCH(B, 3, 3);
MAKE_MATMUL_BENCH(C, 4, 4);
MAKE_MATMUL_BENCH(D, 5, 5);
MAKE_MATMUL_BENCH(E, 6, 6);

MAKE_MATMUL_BENCH(F, 7, 7);
MAKE_MATMUL_BENCH(G, 8, 8);
MAKE_MATMUL_BENCH(H, 9, 9);

MAKE_MATMUL_BENCH(I, 10, 10);
MAKE_MATMUL_BENCH(J, 11, 11);
MAKE_MATMUL_BENCH(K, 12, 12);

MAKE_EINSUM_BENCH(A, 2, 2);
MAKE_EINSUM_BENCH(B, 3, 3);
MAKE_EINSUM_BENCH(C, 4, 4);
MAKE_EINSUM_BENCH(D, 5, 5);
MAKE_EINSUM_BENCH(E, 6, 6);

MAKE_EINSUM_BENCH(F, 7, 7);
MAKE_EINSUM_BENCH(G, 8, 8);
MAKE_EINSUM_BENCH(H, 9, 9);

MAKE_EINSUM_BENCH(I, 10, 10);
MAKE_EINSUM_BENCH(J, 11, 11);

MAKE_EINSUM_BENCH(K, 12, 12);

BENCHMARK_MAIN();