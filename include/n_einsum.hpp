//
// N-operand Einstein summation: "ij,jk,kl->il" with 3 or more tensors.
//
#ifndef EINSTEIN_SUMMATION2_N_EINSUM_HPP
#define EINSTEIN_SUMMATION2_N_EINSUM_HPP
#pragma once
#include "einsum_helpers.hpp"
#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"

using namespace einsum_detail;

template <CNLabels Labels, CNMatrices Matrices> struct NEinsum {
private:
  using value_type = DECAY(Matrices)::value_type;

  // Build per-operand maps label→dim_size. Use make_unique_map_from_sequences
  // because a single operand may have repeated labels (e.g. "ii").
  constexpr static auto all_lmaps = boost::hana::transform(
      boost::hana::zip(DECAY(Labels)::all_input_labels,
                       DECAY(Matrices)::all_extents),
      [](auto pair) {
        return make_unique_map_from_sequences(boost::hana::at_c<0>(pair),
                                              boost::hana::at_c<1>(pair));
      });

  // Combined dimension map: union of all per-operand maps. For duplicate keys
  // hana::union_ keeps the first occurrence; dimensions were already validated
  // at map-construction time (static extents on the mdspans guarantee
  // consistency as long as the user wires the spans correctly — same guarantee
  // as binary Einsum).
  constexpr static auto combined_map = boost::hana::fold_left(
      all_lmaps, boost::hana::make_map(),
      [](auto acc, auto m) { return boost::hana::union_(acc, m); });

public:
  constexpr static auto out_dims =
      boost::hana::transform(DECAY(Labels)::out_labels, [](auto key) {
        return boost::hana::at_key(combined_map, key);
      });
  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  constexpr static auto collapsed_dims =
      boost::hana::transform(DECAY(Labels)::collapsed_labels, [](auto key) {
        return boost::hana::at_key(combined_map, key);
      });
  constexpr static auto collapsed_index_list =
      boost::hana::cartesian_product(make_iota(collapsed_dims));

  constexpr static auto output_iterator_label_map =
      make_output_iterator_label_map(out_index_list, DECAY(Labels)::out_labels);
  constexpr static auto collapsed_iterator_label_map =
      make_output_iterator_label_map(collapsed_index_list,
                                     DECAY(Labels)::collapsed_labels);

  constexpr NEinsum(std::same_as<Labels> auto &&,
                    std::same_as<Matrices> auto &&matrices) noexcept
      : result{}, matrices_{FWD(matrices)} {}

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
  DECAY(Matrices) matrices_;
};

template <CNLabels Labels, CNMatrices Matrices>
NEinsum(Labels &&, Matrices &&) -> NEinsum<Labels, Matrices>;

template <CNLabels Labels, CNMatrices Matrices>
constexpr void NEinsum<Labels, Matrices>::eval() noexcept {
  result.fill(value_type{});
  auto output =
      std::mdspan<value_type, DECAY(decltype(extents))>{result.data(), extents};

  auto accumulate = [&](auto index_map) {
    auto get_index = [&](auto key) {
      return boost::hana::at_key(index_map, key);
    };
    auto out_indices =
        boost::hana::transform(DECAY(Labels)::out_labels, get_index);

    // Fold product: output[out...] += A[a...] * B[b...] * C[c...] * ...
    auto product = boost::hana::fold_left(
        boost::hana::zip(DECAY(Labels)::all_input_labels, matrices_.spans),
        value_type{1}, [&](auto acc, auto pair) {
          auto labels = boost::hana::at_c<0>(pair);
          auto span = boost::hana::at_c<1>(pair); // copy: spans are cheap
          auto indices = boost::hana::transform(labels, get_index);
          return acc * boost::hana::unpack(indices, [&](auto... idx) {
                   return span[idx.value...];
                 });
        });

    boost::hana::unpack(out_indices, [&](auto &&...out_idx) {
      output[out_idx.value...] += product;
    });
  };

  if constexpr (boost::hana::size(DECAY(Labels)::collapsed_labels) == 0) {
    // No contraction: element-wise / outer product across N operands.
    boost::hana::for_each(output_iterator_label_map,
                          [&](auto out_map) { accumulate(out_map); });
  } else if constexpr (boost::hana::size(DECAY(Labels)::out_labels) == 0) {
    // Scalar output: full contraction.
    boost::hana::for_each(collapsed_iterator_label_map,
                          [&](auto col_map) { accumulate(col_map); });
  } else {
    // General case: outer loop over output indices, inner over collapsed.
    boost::hana::for_each(output_iterator_label_map, [&](auto out_map) {
      boost::hana::for_each(collapsed_iterator_label_map, [&](auto col_map) {
        accumulate(boost::hana::union_(out_map, col_map));
      });
    });
  }
}

#define make_einsum_n(name, inputstring, ...)                                  \
  auto name = std::apply(                                                      \
      [](auto... _spans) {                                                     \
        auto m = make_n_matrices(_spans...);                                   \
        auto input = BOOST_HANA_STRING(inputstring);                           \
        auto labels = make_n_label_from_inputs(input);                         \
        return NEinsum{labels, m};                                             \
      },                                                                       \
      std::make_tuple(__VA_ARGS__));

#endif // EINSTEIN_SUMMATION2_N_EINSUM_HPP
