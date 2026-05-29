//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include "printers.hpp"

#include <boost/hana.hpp>
namespace {
template <typename Keys, typename Values>
consteval auto make_map_from_sequences(Keys &&keys, Values &&values) {
  return boost::hana::unpack(
      boost::hana::zip(FWD(keys), FWD(values)), [](auto &&...tuples) {
        return boost::hana::make_map(boost::hana::make_pair(
            boost::hana::at_c<0>(tuples), boost::hana::at_c<1>(tuples))...);
      });
}

template <typename LMap, typename RMap>
consteval bool perform_input_check(LMap &&lmap, RMap &&rmap) {
  auto common_keys = boost::hana::intersection(FWD(lmap), FWD(rmap));
  auto ok = boost::hana::all_of(std::move(common_keys), [&](auto &&key) {
    return lmap[key] == rmap[key];
  });
  return ok;
}

template <typename Extents> consteval auto make_iota(Extents &&extents) {
  auto iotas =
      boost::hana::transform(std::forward<Extents>(extents), [](auto v) {
        return boost::hana::unpack(
            boost::hana::make_range(boost::hana::size_c<0>, v), [](auto... xs) {
              return boost::hana::tuple_c<std::size_t, xs...>;
            });
      });
  return iotas;
}

template <typename ValueList, typename Key>
consteval auto make_output_iterator_label_map(ValueList &&iterator_indices,
                                              Key &&label) {
  auto maps = boost::hana::transform(
      std::forward<ValueList>(iterator_indices), [&](auto &&tup) {
        auto pairs = boost::hana::zip_with(
            [](auto &&k, auto &&v) {
              return boost::hana::make_pair(FWD(k), FWD(v));
            },
            FWD(label), FWD(tup));
        return boost::hana::unpack(std::move(pairs), boost::hana::make_map);
      });
  return maps;
}

template <typename Dims> consteval auto get_extents(Dims &&dims) {
  auto extent = boost::hana::unpack(
      dims, [](auto... dims) { return std::extents<std::size_t, dims...>(); });
  return extent;
}

#define DECAY(x) std::remove_cvref_t<x>
#if NDEBUG
#define PRINT_ITERATION(...)
#else
#define PRINT_ITERATION(lindices, rindices, out_indices)                       \
  print_sequence(out_indices);                                                 \
  std::cout << " += ";                                                         \
  print_sequence(lindices);                                                    \
  std::cout << " * ";                                                          \
  print_sequence(rindices);                                                    \
  std::cout << "\n";
#endif
} // namespace

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

private:
  constexpr static auto output_size = boost::hana::fold_left(
      out_dims, std::size_t{1}, [](auto x, auto y) { return x * y.value; });
  std::array<value_type, output_size> result;
  DECAY(Matrices)::l_matrix_t left;
  DECAY(Matrices)::r_matrix_t right;
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

#define assign_value_macro                                                     \
  auto lindices = boost::hana::transform(DECAY(Labels)::left_labels,           \
                                         get_indices_from_map);                \
  auto rindices = boost::hana::transform(DECAY(Labels)::right_labels,          \
                                         get_indices_from_map);                \
  auto out_indices = boost::hana::values(out_indices_map);                     \
  PRINT_ITERATION(lindices, rindices, out_indices);                            \
  assign_values(lindices, rindices, out_indices)

template <CLabels Labels, CMatrices Matrices>
constexpr void Einsum<Labels, Matrices>::eval() noexcept {
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
  if constexpr (boost::hana::size(DECAY(Labels)::collapsed_labels) == 0) {
    boost::hana::for_each(output_iterator_label_map, [&](auto out_indices_map) {
      auto get_indices_from_map = [&](auto &&key) {
        return *boost::hana::find(out_indices_map, key);
      };
      assign_value_macro;
    });
  } else {
    boost::hana::for_each(output_iterator_label_map, [&](auto out_indices_map) {
      boost::hana::for_each(
          collapsed_iterator_label_map, [&](auto collapsed_indices_map) {
            auto get_indices_from_map = [&](auto &&key) {
              auto val = boost::hana::concat(
                  boost::hana::find(out_indices_map, key),
                  boost::hana::find(collapsed_indices_map, key));
              return *val;
            };
            assign_value_macro;
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
