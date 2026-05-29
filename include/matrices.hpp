//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_MATRICES_HPP
#define EINSTEIN_SUMMATION2_MATRICES_HPP
#pragma once
#include <boost/hana.hpp>
#include <experimental/mdspan>
#include <type_traits>

#define FWD(x) std::forward<decltype(x)>(x)
template <typename T, typename LExt, typename RExt> struct Matrices;

template <typename T, std::size_t... Ls, std::size_t... Rs>
struct Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>> {
  constexpr static auto left_extents =
      boost::hana::make_tuple(boost::hana::size_c<Ls>...);
  constexpr static auto right_extents =
      boost::hana::make_tuple(boost::hana::size_c<Rs>...);

  using value_type = T;
  using l_matrix_t = std::mdspan<T, std::extents<std::size_t, Ls...>>;
  using r_matrix_t = std::mdspan<T, std::extents<std::size_t, Rs...>>;

  l_matrix_t left;
  r_matrix_t right;
  constexpr Matrices(auto &&L_, auto &&R_) noexcept
      : left{FWD(L_)}, right{FWD(R_)} {}
};

template <typename T, std::size_t... Ls, std::size_t... Rs>
Matrices(std::mdspan<T, std::extents<std::size_t, Ls...>>,
         std::mdspan<T, std::extents<std::size_t, Rs...>>)
    -> Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>>;

template <typename> inline constexpr bool is_matrices_v = false;

template <typename T, std::size_t... Ls, std::size_t... Rs>
inline constexpr bool is_matrices_v<
    Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>>> = true;

template <typename M>
concept CMatrices = is_matrices_v<std::remove_cvref_t<M>>;

#endif // EINSTEIN_SUMMATION2_MATRICES_HPP
