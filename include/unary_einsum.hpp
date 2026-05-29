//
// Single-operand Einstein summation: transpose, trace, diagonal, axis
// permutation.
//
#ifndef EINSTEIN_SUMMATION2_UNARY_EINSUM_HPP
#define EINSTEIN_SUMMATION2_UNARY_EINSUM_HPP
#pragma once
#include "einsum_helpers.hpp"
#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"

using namespace einsum_detail;

template <CUniLabels ULabels, CUniMatrices UMatrices> struct UnaryEinsum {
private:
  using value_type = DECAY(UMatrices)::value_type;
  // Use unique-key variant: repeated labels (e.g. "ii") would otherwise create
  // duplicate keys in a Hana map, which is undefined behaviour.
  constexpr static auto lmap = make_unique_map_from_sequences(
      DECAY(ULabels)::left_labels, DECAY(UMatrices)::left_extents);

public:
  constexpr static auto out_dims =
      boost::hana::transform(DECAY(ULabels)::out_labels, [](auto key) {
        return boost::hana::at_key(lmap, key);
      });
  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  constexpr static auto collapsed_dims =
      boost::hana::transform(DECAY(ULabels)::collapsed_labels, [](auto key) {
        return boost::hana::at_key(lmap, key);
      });
  constexpr static auto collapsed_index_list =
      boost::hana::cartesian_product(make_iota(collapsed_dims));

  constexpr static auto output_iterator_label_map =
      make_output_iterator_label_map(out_index_list,
                                     DECAY(ULabels)::out_labels);
  constexpr static auto collapsed_iterator_label_map =
      make_output_iterator_label_map(collapsed_index_list,
                                     DECAY(ULabels)::collapsed_labels);

  constexpr UnaryEinsum(std::same_as<ULabels> auto &&,
                        std::same_as<UMatrices> auto &&matrices) noexcept
      : result{}, left{FWD(matrices).left} {}

  constexpr static auto extents = get_extents(out_dims);
  constexpr void eval() noexcept;
  [[nodiscard]] constexpr auto &get_result() const noexcept { return result; }
  [[nodiscard]] constexpr auto get_result_span() const noexcept {
    return std::mdspan<const value_type,
                       std::remove_const_t<decltype(extents)>>{result.data()};
  }

private:
  constexpr static auto output_size = boost::hana::fold_left(
      out_dims, std::size_t{1}, [](auto x, auto y) { return x * y.value; });
  std::array<value_type, output_size> result;
  DECAY(UMatrices)::l_matrix_t left;
};

template <CUniLabels ULabels, CUniMatrices UMatrices>
UnaryEinsum(ULabels &&, UMatrices &&) -> UnaryEinsum<ULabels, UMatrices>;

template <CUniLabels ULabels, CUniMatrices UMatrices>
constexpr void UnaryEinsum<ULabels, UMatrices>::eval() noexcept {
  result.fill(value_type{});
  auto output =
      std::mdspan<value_type, DECAY(decltype(extents))>{result.data(), extents};

  // Accumulate one index assignment: output[out...] += input[l...]
  // Repeated labels in left_labels (e.g. "ii") correctly map both positions
  // to the same index value, giving diagonal access.
  auto accumulate = [&](auto index_map) {
    auto get_index = [&](auto &&key) {
      return boost::hana::at_key(index_map, key);
    };
    auto lindices =
        boost::hana::transform(DECAY(ULabels)::left_labels, get_index);
    auto out_indices =
        boost::hana::transform(DECAY(ULabels)::out_labels, get_index);
    boost::hana::unpack(out_indices, [&](auto &&...out_idx) {
      boost::hana::unpack(lindices, [&](auto &&...l_idx) {
        output[out_idx.value...] += left[l_idx.value...];
      });
    });
  };

  if constexpr (boost::hana::size(DECAY(ULabels)::collapsed_labels) == 0) {
    // No contraction: permutation, identity, or diagonal extraction.
    boost::hana::for_each(output_iterator_label_map,
                          [&](auto out_map) { accumulate(out_map); });
  } else if constexpr (boost::hana::size(DECAY(ULabels)::out_labels) == 0) {
    // Scalar output: trace "ii->" or full contraction.
    boost::hana::for_each(
        collapsed_iterator_label_map,
        [&](auto collapsed_map) { accumulate(collapsed_map); });
  } else {
    // General unary contraction: e.g. partial trace over some indices.
    boost::hana::for_each(output_iterator_label_map, [&](auto out_map) {
      boost::hana::for_each(
          collapsed_iterator_label_map, [&](auto collapsed_map) {
            accumulate(boost::hana::union_(out_map, collapsed_map));
          });
    });
  }
}

#define make_einsum_unary(name, inputstring, spanA)                            \
  auto name = [_span = spanA]() {                                              \
    UniMatrices m{_span};                                                      \
    auto input = BOOST_HANA_STRING(inputstring);                               \
    auto labels = make_uni_label_from_inputs(input);                           \
    return UnaryEinsum{labels, m};                                             \
  }();

#endif // EINSTEIN_SUMMATION2_UNARY_EINSUM_HPP
