//
// Created by sayan on 9/20/25.
//
#include "einsum_single.hpp"
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
  auto res_einsum = ein.get_result_span();

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
  auto res_einsum = a.get_result_span();

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
  auto res = ein.get_result_span();

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
  auto res = ein.get_result_span();

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
  auto result = ein.get_result_span();

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

  // explicit output required: "ij,ji" auto-infers to scalar (numpy semantics)
  make_einsum(a, "ij,ji->ij", mdA, mdB);
  a.eval();
  auto res = a.get_result_span();

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
  auto res = ein.get_result_span();

  std::vector expected{1, 1, 1, 1, 4, 4, 4, 4, 9, 9, 9, 9, 16, 16, 16, 16};
  std::mdspan<int, std::extents<size_t, 4, 4>> mdexpected{expected.data()};
  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      BOOST_CHECK_EQUAL((res[i, j]), (mdexpected[i, j]));
    }
  }
}

BOOST_AUTO_TEST_CASE(EinsumTest_DotProduct) {
  std::vector A{1, 2, 3};
  std::vector B{4, 5, 6};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  make_einsum(ein, "i,i->", mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_OuterProduct) {
  std::vector A{1, 2, 3};
  std::vector B{4, 5};
  std::mdspan<int, std::extents<size_t, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2>> mdB{B.data()};
  make_einsum(ein, "i,j->ij", mdA, mdB);
  ein.eval();
  auto res = ein.get_result_span();
  std::vector expected{4, 5, 8, 10, 12, 15};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 2; ++j)
      BOOST_CHECK_EQUAL((res[i, j]), (mdexp[i, j]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_MatrixVectorMul) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::vector B{1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3>> mdB{B.data()};
  make_einsum(ein, "ij,j->i", mdA, mdB);
  ein.eval();
  const auto &res = ein.get_result();
  BOOST_CHECK_EQUAL(res[0], 14);
  BOOST_CHECK_EQUAL(res[1], 32);
}

BOOST_AUTO_TEST_CASE(EinsumTest_FrobeniusInnerProduct) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein, "ij,ij->", mdA, mdB);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 70);
}

BOOST_AUTO_TEST_CASE(EinsumTest_EvalIdempotent) {
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein, "ij,jk->ik", mdA, mdB);
  ein.eval();
  auto first = ein.get_result();
  ein.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(first.begin(), first.end(),
                                ein.get_result().begin(),
                                ein.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_AutoInferenceMatchesExplicit) {
  // "ij,jk" auto-infers to "ij,jk->ik" (j repeated → contracted, i and k unique → output)
  std::vector A{1, 2, 3, 4};
  std::vector B{5, 6, 7, 8};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  make_einsum(ein_auto, "ij,jk", mdA, mdB);
  make_einsum(ein_explicit, "ij,jk->ik", mdA, mdB);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ein_auto.get_result().begin(), ein_auto.get_result().end(),
      ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}

// ── Unary einsum tests ────────────────────────────────────────────────────────

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTranspose) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ij->ji", mdA);
  ein.eval();
  auto res = ein.get_result_span();
  std::vector expected{1, 4, 2, 5, 3, 6};
  std::mdspan<int, std::extents<size_t, 3, 2>> mdexp{expected.data()};
  for (auto j = 0; j < 3; ++j)
    for (auto i = 0; i < 2; ++i)
      BOOST_CHECK_EQUAL((res[j, i]), (mdexp[j, i]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryTrace) {
  std::vector A{1, 0, 0, 0, 2, 0, 0, 0, 3};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ii->", mdA);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 6);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryDiagonal) {
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::mdspan<int, std::extents<size_t, 3, 3>> mdA{A.data()};
  make_einsum_unary(ein, "ii->i", mdA);
  ein.eval();
  const auto &res = ein.get_result();
  BOOST_CHECK_EQUAL(res[0], 1);
  BOOST_CHECK_EQUAL(res[1], 5);
  BOOST_CHECK_EQUAL(res[2], 9);
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAxisPermutation) {
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
  std::vector A{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  make_einsum_unary(ein_auto, "ij", mdA);
  make_einsum_unary(ein_explicit, "ij->ij", mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ein_auto.get_result().begin(), ein_auto.get_result().end(),
      ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_UnaryAutoInferTrace) {
  std::vector A{1, 0, 0, 2};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  make_einsum_unary(ein_auto, "ii", mdA);
  make_einsum_unary(ein_explicit, "ii->", mdA);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ein_auto.get_result().begin(), ein_auto.get_result().end(),
      ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryMatMulChain) {
  std::vector A{1, 2, 3, 4, 5, 6};
  std::vector B{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  std::vector C{1, 0, 0, 1, 0, 0, 1, 0};

  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 4, 2>> mdC{C.data()};

  make_einsum(ein, "ij,jk,kl->il", mdA, mdB, mdC);
  ein.eval();
  auto res = ein.get_result_span();

  std::vector expected(4, 0);
  std::mdspan<int, std::extents<size_t, 2, 2>> mdexp{expected.data()};
  for (auto i = 0; i < 2; ++i)
    for (auto l = 0; l < 2; ++l) {
      mdexp[i, l] = 0;
      for (auto j = 0; j < 3; ++j)
        for (auto k = 0; k < 4; ++k)
          mdexp[i, l] += mdA[i, j] * mdB[j, k] * mdC[k, l];
    }
  for (auto i = 0; i < 2; ++i)
    for (auto l = 0; l < 2; ++l)
      BOOST_CHECK_EQUAL((res[i, l]), (mdexp[i, l]));
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryBatchedDot) {
  std::vector v{1, 0, 0, 1};
  std::vector M{1, 0, 0, 1};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdV{v.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdM{M.data()};

  make_einsum(ein, "bi,ij,bj->", mdV, mdM, mdV);
  ein.eval();
  BOOST_CHECK_EQUAL(ein.get_result()[0], 2);
}

BOOST_AUTO_TEST_CASE(EinsumTest_NaryAutoInfer) {
  std::vector A{1, 2, 3, 4};
  std::vector B{1, 0, 0, 1};
  std::vector C{1, 2, 3, 4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdC{C.data()};
  make_einsum(ein_auto, "ij,jk,kl", mdA, mdB, mdC);
  make_einsum(ein_explicit, "ij,jk,kl->il", mdA, mdB, mdC);
  ein_auto.eval();
  ein_explicit.eval();
  BOOST_CHECK_EQUAL_COLLECTIONS(
      ein_auto.get_result().begin(), ein_auto.get_result().end(),
      ein_explicit.get_result().begin(), ein_explicit.get_result().end());
}
