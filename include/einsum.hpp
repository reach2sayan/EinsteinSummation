//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include "einsum_helpers.hpp"
#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include "printers.hpp"

using namespace einsum_detail;

template <CLabels Labels, CMatrices Matrices> struct Einsum {
private:
  using value_type = DECAY(Matrices)::value_type;
  constexpr static auto lmap = make_map_from_sequences(
      DECAY(Labels)::left_labels, DECAY(Matrices)::left_extents);
  constexpr static auto rmap = make_map_from_sequences(
      DECAY(Labels)::right_labels, DECAY(Matrices)::right_extents);
  static_assert(perform_input_check(lmap, rmap), "Input check failed");

public:
  constexpr static auto out_dims =
      boost::hana::transform(DECAY(Labels)::out_labels, [](auto key) {
        return boost::hana::at_key(boost::hana::union_(lmap, rmap), key);
      });
  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  constexpr static auto collapsed_dims =
      boost::hana::transform(DECAY(Labels)::collapsed_labels, [](auto key) {
        return boost::hana::at_key(boost::hana::union_(lmap, rmap), key);
      });
  constexpr static auto collapsed_index_list =
      boost::hana::cartesian_product(make_iota(collapsed_dims));

  constexpr static auto output_iterator_label_map =
      make_output_iterator_label_map(out_index_list, DECAY(Labels)::out_labels);

  constexpr static auto collapsed_iterator_label_map =
      make_output_iterator_label_map(collapsed_index_list,
                                     DECAY(Labels)::collapsed_labels);


  constexpr Einsum(std::same_as<Labels> auto &&,
                   std::same_as<Matrices> auto &&matrices) noexcept
      : result{}, left{FWD(matrices).left}, right{FWD(matrices).right} {}
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
  DECAY(Matrices)::l_matrix_t left;
  DECAY(Matrices)::r_matrix_t right;
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

template <CLabels Labels, CMatrices Matrices>
constexpr void Einsum<Labels, Matrices>::eval() noexcept {
  result.fill(value_type{});
  auto output =
      std::mdspan<value_type, DECAY(decltype(extents))>{result.data(), extents};

  auto assign_values = [&](auto &&lindices, auto &&rindices,
                           auto &&out_indices) {
    boost::hana::unpack(out_indices, [&](auto &&...out_idx) {
      boost::hana::unpack(lindices, [&](auto &&...l_idx) {
        boost::hana::unpack(rindices, [&](auto &&...r_idx) {
          output[out_idx.value...] +=
              left[l_idx.value...] * right[r_idx.value...];
        });
      });
    });
  };

  auto accumulate = [&](auto index_map) {
    auto get_index = [&](auto &&key) {
      return boost::hana::at_key(index_map, key);
    };
    auto lindices = boost::hana::transform(DECAY(Labels)::left_labels, get_index);
    auto rindices = boost::hana::transform(DECAY(Labels)::right_labels, get_index);
    auto out_indices = boost::hana::transform(DECAY(Labels)::out_labels, get_index);
    PRINT_ITERATION(lindices, rindices, out_indices);
    assign_values(lindices, rindices, out_indices);
  };

  if constexpr (boost::hana::size(DECAY(Labels)::collapsed_labels) == 0) {
    // no contraction: element-wise or outer product
    boost::hana::for_each(output_iterator_label_map, [&](auto out_map) {
      accumulate(out_map);
    });
  } else if constexpr (boost::hana::size(DECAY(Labels)::out_labels) == 0) {
    // scalar output (e.g. "i,i->" dot product): iterate only collapsed indices
    boost::hana::for_each(collapsed_iterator_label_map, [&](auto collapsed_map) {
      accumulate(collapsed_map);
    });
  } else {
    // general contraction: outer = output indices, inner = collapsed indices
    boost::hana::for_each(output_iterator_label_map, [&](auto out_map) {
      boost::hana::for_each(
          collapsed_iterator_label_map, [&](auto collapsed_map) {
            accumulate(boost::hana::union_(out_map, collapsed_map));
          });
    });
  }
}

#define make_einsum(name, inputstring, spanA, spanB)                           \
  auto name = [_lhs = spanA, _rhs = spanB]() {                                \
    Matrices m{_lhs, _rhs};                                                    \
    auto input = BOOST_HANA_STRING(inputstring);                               \
    auto labels = make_label_from_inputs(input);                               \
    return Einsum{labels, m};                                                  \
  }();

#endif // EINSTEIN_SUMMATION2_EINSUM_2_HPP
