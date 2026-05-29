//
// Created by sayan on 9/20/25.
//
#include "einsum.hpp"
#include "unary_einsum.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include <algorithm>
#include <list>
#include <ranges>
#include <vector>

#define BOOST_TEST_MODULE EinsumTestSuite
#include <boost/test/included/unit_test.hpp>

// Helper: wrap a get_result() array in a const mdspan using the einsum's
// own static extents, so multi-index [i,j,...] keeps working.
#define RESULT_VIEW(ein, name)                                                 \
  const auto &name##_data = ein.get_result();                                 \
  std::mdspan<const int, std::remove_const_t<decltype(decltype(ein)::extents)>> name { \
    name##_data.data()                                                         \
  }

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix1) {
  std::vector A{0, 1, 2, 3, 4, 5};
  std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  make_einsum(ein, "ij,jk->ik", mdA, mdB);
  ein.eval();
  RESULT_VIEW(ein, res_einsum);

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
      BOOST_CHECK_EQUAL((mdres[i, j]), (res_einsum[i, j]));
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
  RESULT_VIEW(a, res_einsum);

  std::vector res{0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdres{res.data()};
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
      BOOST_CHECK_EQUAL((mdres[i, j]), (res_einsum[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixMul) {
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  make_einsum(ein, "ij,jk->ik", mdmat1, mdmat2);
  ein.eval();
  auto res = ein.get_result_span();  // exercises get_result_span() on binary path

  std::vector res_calc{130, 130, 130, 130, 230, 230, 230, 230,
                       330, 330, 330, 330, 430, 430, 430, 430};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct) {
  std::vector mat1{11, 12, 13, 14, 21, 22, 23, 24,
                   31, 32, 33, 34, 41, 42, 43, 44};
  std::vector mat2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

  make_einsum(ein, "ij,ij->ij", mdmat1, mdmat2);
  ein.eval();
  RESULT_VIEW(ein, res);

  std::vector res_calc{11, 12, 13, 14,  42,  44,  46,  48,
                       93, 96, 99, 102, 164, 168, 172, 176};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_HadamardProduct2) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::vector res_calc{5, 12, 21, 32};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};

  make_einsum(ein, "ij,ij->ij", mdA, mdB);
  ein.eval();
  RESULT_VIEW(ein, result);

  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      BOOST_CHECK_EQUAL((result[i, j]), (mdmatres[i, j]));
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
  RESULT_VIEW(a, res);

  // "ij,ji->ij": element-wise product of A with transpose(B)
  // result[i,j] = A[i,j] * B[j,i]
  std::vector res_calc{1, 6, 6, 16};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdmatres{res_calc.data()};
  for (auto i = 0; i < 2; i++) {
    for (auto j = 0; j < 2; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdmatres[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_ElementWiseSquaring) {
  std::vector mat{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdmat{mat.data()};

  make_einsum(ein, "ij,ij->ij", mdmat, mdmat);
  ein.eval();
  RESULT_VIEW(ein, res);

  std::vector expected{1, 1, 1, 1, 4, 4, 4, 4, 9, 9, 9, 9, 16, 16, 16, 16};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdexpected{expected.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdexpected[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_DotProduct) {
  // "i,i->" — vector dot product → scalar
  std::vector A{1, 2, 3};
  std::vector B{4, 5, 6};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  make_einsum(ein, "i,i->", mdA, mdB);
  ein.eval();
  // 1*4 + 2*5 + 3*6 = 32
  BOOST_CHECK_EQUAL(ein.get_result()[0], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_OuterProduct) {
  // "i,j->ij" — outer product of two vectors → matrix
  std::vector A{1, 2, 3};
  std::vector B{4, 5};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2>> mdB{B.data()};
  make_einsum(ein, "i,j->ij", mdA, mdB);
  ein.eval();
  RESULT_VIEW(ein, res);
  // res[i,j] = A[i] * B[j]
  std::vector expected{4, 5, 8, 10, 12, 15};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 2; ++j)
      BOOST_CHECK_EQUAL((res[i, j]), (mdexp[i, j]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixVectorMul) {
  // "ij,j->i" — matrix × vector
  std::vector A{1, 2, 3, 4, 5, 6};
  std::vector B{1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  make_einsum(ein, "ij,j->i", mdA, mdB);
  ein.eval();
  const auto &res = ein.get_result();
  // res[0] = 1*1+2*2+3*3 = 14, res[1] = 4*1+5*2+6*3 = 32
  BOOST_CHECK_EQUAL(res[0], 14);
  BOOST_CHECK_EQUAL(res[1], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_FrobeniusInnerProduct) {
  // "ij,ij->" — full contraction of two matrices → scalar
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein, "ij,ij->", mdA, mdB);
  ein.eval();
  // 1*5 + 2*6 + 3*7 + 4*8 = 70
  BOOST_CHECK_EQUAL(ein.get_result()[0], 70);
}

BOOST_AUTO_TEST_CASE(EinsumTest_EvalIdempotent) {
  // eval() must reset the result each call — not accumulate across calls
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein, "ij,jk->ik", mdA, mdB);
  ein.eval();
  auto first = ein.get_result();
  ein.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(first.begin(), first.end(),
                                ein.get_result().begin(), ein.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_AutoInferenceMatchesExplicit) {
  // "ij,ji" auto-infers to "ij,ji->ij" — result must match the explicit form
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein_auto, "ij,ji", mdA, mdB);
  make_einsum(ein_explicit, "ij,ji->ij", mdA, mdB);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}

// ── Unary einsum tests ────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTranspose) {
  // "ij->ji": result[j,i] == A[i,j]
  std::vector A{1, 2, 3, 4, 5, 6};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ij->ji", mdA);
  ein.eval();
  auto res = ein.get_result_span();
  // expected: [[1,4],[2,5],[3,6]]
  std::vector expected{1, 4, 2, 5, 3, 6};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto j = 0; j < 3; ++j)
    for (auto i = 0; i < 2; ++i)
      BOOST_CHECK_EQUAL((res[j, i]), (mdexp[j, i]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTrace) {
  // "ii->": sum of diagonal of a 3×3 matrix
  std::vector A{1, 0, 0,
                0, 2, 0,
                0, 0, 3};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ii->", mdA);
  ein.eval();
  // trace = 1+2+3 = 6
  BOOST_CHECK_EQUAL(ein.get_result()[0], 6);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryDiagonal) {
  // "ii->i": extract diagonal of a 3×3 matrix
  std::vector A{1, 2, 3,
                4, 5, 6,
                7, 8, 9};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ii->i", mdA);
  ein.eval();
  const auto &res = ein.get_result();
  // diagonal: [1, 5, 9]
  BOOST_CHECK_EQUAL(res[0], 1);
  BOOST_CHECK_EQUAL(res[1], 5);
  BOOST_CHECK_EQUAL(res[2], 9);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAxisPermutation) {
  // "ijk->kij": result[k,i,j] == A[i,j,k] on a 2×2×2 tensor
  // A[i,j,k] = i*4 + j*2 + k + 1  (values 1..8)
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2, 2>> mdA{A.data()};
  make_einsum_unary(ein, "ijk->kij", mdA);
  ein.eval();
  auto res = ein.get_result_span();
  for (auto i = 0; i < 2; ++i)
    for (auto j = 0; j < 2; ++j)
      for (auto k = 0; k < 2; ++k)
        BOOST_CHECK_EQUAL((res[k, i, j]), (mdA[i, j, k]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferIdentity) {
  // "ij" auto-infers to "ij->ij" for a single operand
  std::vector A{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  make_einsum_unary(ein_auto,     "ij",    mdA);
  make_einsum_unary(ein_explicit, "ij->ij", mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferTrace) {
  // "ii" auto-infers to "ii->" (repeated label → trace)
  std::vector A{1, 0, 0, 2};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  make_einsum_unary(ein_auto,     "ii",   mdA);
  make_einsum_unary(ein_explicit, "ii->", mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(ein_auto.get_result().begin(), ein_auto.get_result().end(),
                                ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}
