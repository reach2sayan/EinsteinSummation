[![CMake](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml/badge.svg)](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml) [![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
# Einstein Summation (einsum)

A C++23 header-only implementation of the Einstein summation convention for tensor operations, similar to NumPy's `einsum`. Index labels, output shape, and contraction axes are all resolved at compile time using [Boost.Hana](https://github.com/boostorg/hana) and `std::mdspan`.

## Features

- Compact tensor operation notation using Einstein summation convention
- Arbitrary tensor rank and dimension sizes (static extents only)
- Dimension compatibility checked at compile time via `static_assert`
- Automatic output shape inference when the `->` clause is omitted
- Zero runtime overhead for index bookkeeping — all iteration spaces are `constexpr`
- Single-operand operations: transpose, trace, diagonal extraction, axis permutation
- N-operand contractions: chain `"ij,jk,kl->il"` with 3 or more tensors
- `get_result_span()` returns a typed `std::mdspan` for direct multi-dimensional access

## Requirements

- C++23 compiler — tested with GCC 13 and Clang 18
- CMake ≥ 3.14
- Boost (headers only — Hana)
- [`kokkos/mdspan`](https://github.com/kokkos/mdspan) (fetched automatically via CMake FetchContent)

## Usage

Include `einsum.hpp` / `unary_einsum.hpp` / `n_einsum.hpp` and use the corresponding macro, then call `eval()`.  
`get_result_span()` returns a typed `std::mdspan` owned by the einsum object — keep the object alive while using the span.

### Matrix multiplication

```cpp
std::vector A{0, 1, 2, 3, 4, 5};
std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};

make_einsum(ein, "ij,jk->ik", mdA, mdB);
ein.eval();
auto res = ein.get_result_span(); // std::mdspan<const int, extents<size_t, 2, 4>>
```

### Hadamard product (element-wise multiplication)

```cpp
std::vector A{1, 2, 3, 4};
std::vector B{5, 6, 7, 8};

std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

make_einsum(ein, "ij,ij->ij", mdA, mdB);
ein.eval();
auto res = ein.get_result_span(); // [5, 12, 21, 32]
```

### Tensor contraction

```cpp
std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdB{B.data()};

// Sum over 'w' and 'h', keeping 'b', 'i', 'j'
make_einsum(ein, "bhwi,bhwj->bij", mdA, mdB);
ein.eval();
```

### Automatic output shape inference

If the `->` clause is omitted, the library infers the output indices as the sorted union of all non-repeated input indices:

```cpp
make_einsum(ein, "ij,jk", mdA, mdB); // equivalent to "ij,jk->ik"
ein.eval();
```

### N-operand contractions (3 or more tensors)

```cpp
#include "n_einsum.hpp"

// 3-operand matrix chain: A(2×3) × B(3×4) × C(4×2) → result(2×2)
make_einsum_n(ein, "ij,jk,kl->il", mdA, mdB, mdC);
ein.eval();
auto res = ein.get_result_span();
```

### Single-operand operations

```cpp
#include "unary_einsum.hpp"

// Transpose: result[j,i] = A[i,j]
make_einsum_unary(ein, "ij->ji", mdA);
ein.eval();
auto res = ein.get_result_span();

// Trace: sum of diagonal → scalar
make_einsum_unary(ein, "ii->", mdA);
ein.eval();
int trace = ein.get_result()[0];

// Diagonal extraction: result[i] = A[i,i]
make_einsum_unary(ein, "ii->i", mdA);
ein.eval();

// Axis permutation: result[k,i,j] = A[i,j,k]
make_einsum_unary(ein, "ijk->kij", mdA);
ein.eval();
```

### Not yet supported

- Dynamic (runtime) extents
