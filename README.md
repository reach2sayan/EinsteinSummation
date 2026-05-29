[![CMake](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml/badge.svg)](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml) [![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
# Einstein Summation (einsum)

A C++23 header-only implementation of the Einstein summation convention for tensor operations, similar to NumPy's `einsum`. Index labels, output shape, and contraction axes are all resolved at compile time using [Boost.Hana](https://github.com/boostorg/hana) and `std::mdspan`.

## Features

- Compact tensor operation notation using Einstein summation convention
- Arbitrary tensor rank and dimension sizes (static extents only)
- Dimension compatibility checked at compile time via `static_assert`
- Automatic output shape inference when the `->` clause is omitted
- Zero runtime overhead for index bookkeeping — all iteration spaces are `constexpr`

## Requirements

- C++23 compiler — tested with GCC 13 and Clang 18
- CMake ≥ 3.14
- Boost (headers only — Hana)
- [`kokkos/mdspan`](https://github.com/kokkos/mdspan) (fetched automatically via CMake FetchContent)

## Usage

Include `einsum.hpp` and use the `make_einsum` macro to declare an operation, then call `eval()`.  
`get_result()` returns a `const std::array<T, N>&` owned by the einsum object — keep the object alive while using the reference.

### Matrix multiplication

```cpp
std::vector A{0, 1, 2, 3, 4, 5};
std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};

make_einsum(ein, "ij,jk->ik", mdA, mdB);
ein.eval();
const auto& result = ein.get_result(); // const std::array<int, 8>&
```

### Hadamard product (element-wise multiplication)

```cpp
std::vector A{1, 2, 3, 4};
std::vector B{5, 6, 7, 8};

std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

make_einsum(ein, "ij,ij->ij", mdA, mdB);
ein.eval();
const auto& result = ein.get_result(); // [5, 12, 21, 32]
```

### Element-wise squaring

```cpp
std::vector mat{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
std::mdspan<int, std::extents<size_t, 4, 4>> mdmat{mat.data()};

make_einsum(ein, "ij,ij->ij", mdmat, mdmat);
ein.eval();
const auto& result = ein.get_result(); // [1,1,1,1, 4,4,4,4, 9,9,9,9, 16,16,16,16]
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

### Element-wise product with transposed operand

```cpp
std::vector A{1, 2, 3, 4};
std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2>> mdB{A.data()};

// result[i,j] = A[i,j] * B[j,i]
make_einsum(ein, "ij,ji->ij", mdA, mdB);
ein.eval();
const auto& result = ein.get_result(); // [1, 6, 6, 16]
```

### Not yet supported

- Dynamic (runtime) extents
- More than two input tensors
- Single-operand index permutation (true transpose: `"ij->ji"`)
