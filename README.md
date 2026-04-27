[![CMake](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml/badge.svg)](https://github.com/reach2sayan/Einstein_Summation/actions/workflows/action.yml) [![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
# Einstein Summation (einsum)

A C++26 implementation of the Einstein summation convention for tensor operations. This library allows for concise
notation of tensor operations through labeled indices, similar to NumPy's einsum function in Python.

Uses [Boost Hana](https://github.com/boostorg/hana) extensibly 

## Overview

Einstein summation is a powerful notation for expressing operations on multi-dimensional arrays (tensors). This library
implements this notation in C++23, leveraging modern features like , compile-time programming, and metaprogramming to
provide efficient tensor operations with a clean interface. `std::mdspan`

## Features

- Compact tensor operation notation using Einstein summation convention
- Support for arbitrary dimensions and tensor ranks
- Static dimension checking at compile time
- Full support for C++23 features including `std::mdspan`
- Template metaprogramming for compile-time optimization
- Automatic result shape inference

## Requirements

- C++23 compatible compiler (tested with GCC)
- CMake build system
- Support for experimental features () `std::mdspan`

## Usage

### Basic Matrix Multiplication

``` cpp
// Create input data
std::vector A{0, 1, 2, 3, 4, 5};
std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// Create mdspans with appropriate dimensions
std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};

// Perform matrix multiplication using einsum
// "ij,jk->ik" represents matrix multiplication in Einstein notation
make_einsum(ein, "ij,jk->ik", mdA, mdB);
ein.eval();
auto result = ein.get_result();

// Result is now a 2x4 matrix
```

### Hadamard Product (Element-wise Multiplication)

The Hadamard product performs element-wise multiplication between tensors of the same shape. Using Einstein notation,
this operation can be expressed by using the same indices in both input tensors and the output:

``` cpp
// Create input data
std::vector A{1, 2, 3, 4};
std::vector B{5, 6, 7, 8};

// Create mdspans with identical dimensions
std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

// Perform element-wise multiplication (Hadamard product)
// "ij,ij->ij" represents element-wise multiplication in Einstein notation
make_einsum(ein, "ij,ij->ij", mdA, mdB);
ein.eval();
auto result = ein.get_result();

// Result contains element-wise products: [5, 12, 21, 32]
```

### Batch Operations with Tensor Contraction

``` cpp
std::vector A{/* data */};
std::vector B{/* data */};
std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdA{A.data()};
std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdB{B.data()};

// Sum over 'w' and 'h' dimensions, keeping batch 'b', input 'i', and output 'j'
make_einsum(result,"bhwi,bhwj->"bij", mdA, mdB);
result.eval();
```

### Matrix Transpose [[Not Supported Yet]]

Matrix transpose can be elegantly expressed using einsum by swapping the indices in the output:

```cpp 
std::vector A{1, 2, 3, 4}; 
std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A. data()};
// Empty first operand, second operand with "ij", result with swapped indices "ji" 
auto a = seinsum("ij", "ji", mdA); 
a.eval(); 
auto result = a.get_result();
// Result is the transpose of the input matrix: [1, 3, 2, 4]
```


### Element-wise Squaring

Einsum can also be used for element-wise operations on a single tensor by using it twice with identical indices:

```cpp 
std::vector mat{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}; 
std::mdspan<int, std::extents<size_t, 4, 4>> mdmat{mat.data()};
// Element-wise squaring using the same tensor twice 
make_einsum(ein, "ij,ij->ij", mdmat, mdmat); 
ein.eval(); 
auto result = ein.get_result();
// Result contains squared values: [1, 1, 1, 1, 4, 4, 4, 4, 9, 9, 9, 9, 16, 16, 16, 16]
```


### Automatic Result Shape Inference

If you omit the result labels, the library will automatically determine the appropriate result shape:

``` cpp
auto result = auto_einsum("bhwi", "bhwj", mdA, mdB);
result.eval();
```
