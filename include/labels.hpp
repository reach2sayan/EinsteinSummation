//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_MATRIX_HPP
#define EINSTEIN_SUMMATION2_MATRIX_HPP
#pragma once
#include <boost/hana.hpp>

namespace einsum_detail {
template <typename LLabels, typename RLabels, typename OutLabels>
consteval auto make_collapsed_labels(LLabels ll, RLabels rl, OutLabels ol) {
  auto sorted_input_labels = boost::hana::sort(boost::hana::concat(ll, rl));
  auto diff = boost::hana::filter(sorted_input_labels, [&](auto l) {
    return boost::hana::not_(boost::hana::contains(ol, l));
  });
  return boost::hana::unique(diff);
}
} // namespace einsum_detail

template <typename LSeq, typename RSeq, typename OutSeq> struct Labels {
  static_assert(false);
};

template <char... Ls, char... Rs, char... Out>
struct Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
              boost::hana::string<Out...>> {
  constexpr static auto left_labels =
      boost::hana::make_tuple(boost::hana::char_c<Ls>...);
  constexpr static auto right_labels =
      boost::hana::make_tuple(boost::hana::char_c<Rs>...);
  constexpr static auto out_labels =
      boost::hana::make_tuple(boost::hana::char_c<Out>...);
  constexpr static auto collapsed_labels = einsum_detail::make_collapsed_labels(
      left_labels, right_labels, out_labels);
};

template <char... Ls, char... Rs, char... Out>
consteval auto make_labels(boost::hana::string<Ls...>,
                           boost::hana::string<Rs...>,
                           boost::hana::string<Out...>) {
  return Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
                boost::hana::string<Out...>>{};
}

template <typename> inline constexpr bool is_labels_v = false;
template <char... Ls, char... Rs, char... Out>
inline constexpr bool
    is_labels_v<Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
                       boost::hana::string<Out...>>> = true;

template <typename T>
concept CLabels = is_labels_v<std::remove_cvref_t<T>>;

template <typename LSeq, typename OutSeq> struct UniLabels {
  static_assert(false);
};

template <char... Ls, char... Out>
struct UniLabels<boost::hana::string<Ls...>, boost::hana::string<Out...>> {
  constexpr static auto left_labels =
      boost::hana::make_tuple(boost::hana::char_c<Ls>...);
  constexpr static auto out_labels =
      boost::hana::make_tuple(boost::hana::char_c<Out>...);
  constexpr static auto collapsed_labels = einsum_detail::make_collapsed_labels(
      left_labels, boost::hana::make_tuple(), out_labels);
};

template <char... Ls, char... Out>
consteval auto make_uni_labels(boost::hana::string<Ls...>,
                               boost::hana::string<Out...>) {
  return UniLabels<boost::hana::string<Ls...>, boost::hana::string<Out...>>{};
}

template <typename> inline constexpr bool is_uni_labels_v = false;
template <char... Ls, char... Out>
inline constexpr bool is_uni_labels_v<
    UniLabels<boost::hana::string<Ls...>, boost::hana::string<Out...>>> = true;

template <typename T>
concept CUniLabels = is_uni_labels_v<std::remove_cvref_t<T>>;

template <typename Out, typename... Inputs> struct NLabels {
  constexpr static auto out_labels =
      boost::hana::unpack(Out{}, boost::hana::make_tuple);
  constexpr static auto all_input_labels = boost::hana::make_tuple(
      boost::hana::unpack(Inputs{}, boost::hana::make_tuple)...);

  constexpr static auto all_labels_flat = boost::hana::fold_left(
      all_input_labels, boost::hana::make_tuple(),
      [](auto acc, auto labels) { return boost::hana::concat(acc, labels); });

  constexpr static auto collapsed_labels = boost::hana::unique(
      boost::hana::sort(boost::hana::filter(all_labels_flat, [](auto l) {
        return boost::hana::not_(boost::hana::contains(out_labels, l));
      })));
};

template <typename Out, typename... Inputs>
consteval auto make_n_labels(Out, Inputs...) {
  return NLabels<Out, Inputs...>{};
}

template <typename> inline constexpr bool is_n_labels_v = false;
template <typename Out, typename... Inputs>
inline constexpr bool is_n_labels_v<NLabels<Out, Inputs...>> = true;

template <typename T>
concept CNLabels = is_n_labels_v<std::remove_cvref_t<T>>;

#endif // EINSTEIN_SUMMATION2_MATRIX_HPP
