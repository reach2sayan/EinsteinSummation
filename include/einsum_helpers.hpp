//
// Shared compile-time helpers used by both einsum.hpp and unary_einsum.hpp.
//
#ifndef EINSTEIN_SUMMATION2_EINSUM_HELPERS_HPP
#define EINSTEIN_SUMMATION2_EINSUM_HELPERS_HPP
#pragma once
#include <boost/hana.hpp>
#include <experimental/mdspan>

#define DECAY(x) std::remove_cvref_t<x>
#define FWD_H(x) std::forward<decltype(x)>(x)

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

namespace einsum_detail {

template <typename Keys, typename Values>
consteval auto make_map_from_sequences(Keys &&keys, Values &&values) {
  return boost::hana::unpack(
      boost::hana::zip(FWD_H(keys), FWD_H(values)), [](auto &&...tuples) {
        return boost::hana::make_map(boost::hana::make_pair(
            boost::hana::at_c<0>(tuples), boost::hana::at_c<1>(tuples))...);
      });
}

// Like make_map_from_sequences but silently drops duplicate keys (first wins).
// Required for unary einsum with repeated labels, e.g. "ii->" where left_labels
// contains 'i' twice — Hana maps require unique keys.
template <typename Keys, typename Values>
consteval auto make_unique_map_from_sequences(Keys keys, Values values) {
  return boost::hana::fold_left(
      boost::hana::zip(std::move(keys), std::move(values)),
      boost::hana::make_map(),
      [](auto acc, auto pair) {
        return boost::hana::if_(
            boost::hana::contains(acc, boost::hana::at_c<0>(pair)),
            acc,
            boost::hana::insert(acc, boost::hana::make_pair(
                boost::hana::at_c<0>(pair), boost::hana::at_c<1>(pair))));
      });
}

template <typename LMap, typename RMap>
consteval bool perform_input_check(LMap &&lmap, RMap &&rmap) {
  auto common_keys = boost::hana::intersection(FWD_H(lmap), FWD_H(rmap));
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
              return boost::hana::make_pair(FWD_H(k), FWD_H(v));
            },
            FWD_H(label), FWD_H(tup));
        return boost::hana::unpack(std::move(pairs), boost::hana::make_map);
      });
  return maps;
}

template <typename Dims> consteval auto get_extents(Dims &&dims) {
  return boost::hana::unpack(
      std::forward<Dims>(dims),
      [](auto... d) { return std::extents<std::size_t, d...>(); });
}

} // namespace einsum_detail

#endif // EINSTEIN_SUMMATION2_EINSUM_HELPERS_HPP
