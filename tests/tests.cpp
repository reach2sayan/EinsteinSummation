//
// Created by sayan on 9/20/25.
//
#include "einsum.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include <algorithm>
#include <list>
#include <ranges>
#include <vector>

#define BOOST_TEST_MODULE EinsumTestSuite
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix1) {
  std::vector A{0, 1, 2, 3, 4, 5};
  std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  make_einsum(ein, "ij,jk->ik", mdA, mdB);
  ein.eval();
  auto res_einsum = ein.get_result();

  std::vector res{0, 0, 0, 0, 0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 4; ++k) {
      mdres[i, k] = 0;
      for (auto j = 0; j < 3; ++j) {
        mdres[i, k] += mdA[i, j] * mdB[j, k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 4; ++j) {
      auto v1 = mdres[i, j];
      auto v2 = res_einsum[i, j];
      BOOST_CHECK_EQUAL(v1, v2);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix2) {
  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  make_einsum(a, "ij,jk->ik", mdA, mdB);
  a.eval();
  auto res_einsum = a.get_result();

  std::vector res{0, 0, 0, 0, 0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 2; ++k) {
      mdres[i, k] = 0;
      for (auto j = 0; j < 2; ++j) {
        mdres[i, k] += mdA[i, j] * mdB[j, k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      auto v1 = mdres[i, j];
      auto v2 = res_einsum[i, j];
      BOOST_CHECK_EQUAL(v1, v2);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixMul) {
  std::vector vec{0, 1, 2, 3};
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  make_einsum(ein, "ij,jk->ik", mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result();
  std::vector res_calc{130, 130, 130, 130, 230, 230, 230, 230,
                       330, 330, 330, 330, 430, 430, 430, 430};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      auto l = res[i, j];
      auto r = mdmatres[i, j];
      BOOST_CHECK_EQUAL(l, r);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct) {
  std::vector vec{0, 1, 2, 3};
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};
  make_einsum(ein, "ij,ij->ij", mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result();
  std::vector res_calc{11, 12, 13, 14,  42,  44,  46,  48,
                       93, 96, 99, 102, 164, 168, 172, 176};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      auto l = res[i, j];
      auto r = mdmatres[i, j];
      BOOST_CHECK_EQUAL(l, r);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct2) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::vector res_calc{5, 12, 21, 32};

  // Create mdspans with identical dimensions
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};
  make_einsum(ein, "ij,ij->ij", mdA, mdB);
  ein.eval();
  auto result = ein.get_result();

  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      auto l = result[i, j];
      auto r = mdmatres[i, j];
      BOOST_CHECK_EQUAL(l, r);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixTranspose) {
  std::vector A{1, 2, 3, 4};
  std::vector<int> B{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{A.data()};

  make_einsum(a, "ij,ji", mdA, mdB);
  a.eval();
  auto res = a.get_result();

  std::vector res_calc{1, 3, 2, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      auto r = res[i, j];
      auto l = mdmatres[i, j];
      BOOST_CHECK_EQUAL(l, r);
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_ElementWiseSquaring) {
  std::vector vec{0, 1, 2, 3};
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};
  mdmat1 = mdmat2;
  make_einsum(ein, "ij,ij->ij", mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result();
  // Add some actual verification for the element-wise squaring
  std::vector expected{1, 1, 1, 1, 4, 4, 4, 4, 9, 9, 9, 9, 16, 16, 16, 16};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdexpected{expected.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      auto r = res[i, j];
      auto exp = mdexpected[i, j];
      BOOST_CHECK_EQUAL(r, exp);
    }
  }
}
