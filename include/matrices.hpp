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

template <typename T, typename LExt> struct UniMatrices;

template <typename T, std::size_t... Ls>
struct UniMatrices<T, std::index_sequence<Ls...>> {
  constexpr static auto left_extents =
      boost::hana::make_tuple(boost::hana::size_c<Ls>...);

  using value_type = T;
  using l_matrix_t = std::mdspan<T, std::extents<std::size_t, Ls...>>;

  l_matrix_t left;
  constexpr explicit UniMatrices(auto &&L_) noexcept : left{FWD(L_)} {}
};

template <typename T, std::size_t... Ls>
UniMatrices(std::mdspan<T, std::extents<std::size_t, Ls...>>)
    -> UniMatrices<T, std::index_sequence<Ls...>>;

template <typename> inline constexpr bool is_uni_matrices_v = false;
template <typename T, std::size_t... Ls>
inline constexpr bool
    is_uni_matrices_v<UniMatrices<T, std::index_sequence<Ls...>>> = true;

template <typename M>
concept CUniMatrices = is_uni_matrices_v<std::remove_cvref_t<M>>;

// Convert index_sequence<Ls...> to std::extents<size_t, Ls...>.
template <typename Seq> struct seq_to_extents;
template <std::size_t... Ls> struct seq_to_extents<std::index_sequence<Ls...>> {
  using type = std::extents<std::size_t, Ls...>;
};

// Convert index_sequence<Ls...> to a hana tuple of size constants.
template <std::size_t... Ls>
consteval auto seq_to_hana_extents(std::index_sequence<Ls...>) {
  return boost::hana::make_tuple(boost::hana::size_c<Ls>...);
}

template <typename T, typename... Seqs> struct NMatrices {
  using value_type = T;
  using spans_type = boost::hana::tuple<
      std::mdspan<T, typename seq_to_extents<Seqs>::type>...>;

  // Per-operand extent tuples for compile-time lmap building.
  constexpr static auto all_extents =
      boost::hana::make_tuple(seq_to_hana_extents(Seqs{})...);

  spans_type spans;
  constexpr explicit NMatrices(
      std::mdspan<T, typename seq_to_extents<Seqs>::type>... s) noexcept
      : spans{s...} {}
};

// Trait: extract index_sequence from an mdspan type.
template <typename Span> struct span_to_seq;
template <typename T, std::size_t... Ls>
struct span_to_seq<std::mdspan<T, std::extents<std::size_t, Ls...>>> {
  using type = std::index_sequence<Ls...>;
  using value_type = T;
};

// Factory deducing NMatrices<T, Seq1, Seq2, ...> from N mdspan arguments.
template <typename... Spans> constexpr auto make_n_matrices(Spans... spans) {
  using T = typename span_to_seq<
      std::tuple_element_t<0, std::tuple<Spans...>>>::value_type;
  return NMatrices<T, typename span_to_seq<Spans>::type...>{spans...};
}

template <typename> inline constexpr bool is_n_matrices_v = false;
template <typename T, typename... Seqs>
inline constexpr bool is_n_matrices_v<NMatrices<T, Seqs...>> = true;

template <typename M>
concept CNMatrices = is_n_matrices_v<std::remove_cvref_t<M>>;

#endif // EINSTEIN_SUMMATION2_MATRICES_HPP
