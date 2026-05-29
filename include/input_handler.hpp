//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_HELPERS_HPP
#define EINSTEIN_SUMMATION2_HELPERS_HPP
#pragma once
#include "labels.hpp"
#include <boost/hana.hpp>
#include <tuple>

namespace einsum_detail {

consteval auto tuple_to_string(auto &&tuple) {
  return boost::hana::unpack(tuple, [](auto &&...cs) {
    return boost::hana::string<std::decay_t<decltype(cs)>::value...>{};
  });
}

consteval auto stable_unique(auto &&xs) {
  return boost::hana::fold_left(
      xs, boost::hana::make_tuple(), [](auto acc, auto x) {
        return boost::hana::if_(boost::hana::contains(acc, x), acc,
                                boost::hana::append(acc, x));
      });
}

consteval auto parse_input(auto &&input_string) {
  auto [l, r, out] = boost::hana::unpack(input_string, [](auto &&...chars) {
    auto input = boost::hana::make_tuple(chars...);

    constexpr auto pos_comma = boost::hana::index_if(
        input, [](auto &&c) { return c == boost::hana::char_c<','>; });
    auto ls = boost::hana::take_front(input, *pos_comma);

    auto rest =
        boost::hana::drop_front(input, *pos_comma + boost::hana::size_c<1>);

    constexpr auto pos_arrow = boost::hana::index_if(
        rest, [](auto &&c) { return c == boost::hana::char_c<'-'>; });
    auto rs = boost::hana::take_front(rest, *pos_arrow);

    auto final =
        boost::hana::drop_front(rest, *pos_arrow + boost::hana::size_c<2>);

    return std::make_tuple(tuple_to_string(ls), tuple_to_string(rs),
                           tuple_to_string(final));
  });

  return boost::hana::make_tuple(std::move(l), std::move(r), std::move(out));
}

consteval auto make_label_from_inputs(auto input_string) {
  if constexpr (!boost::hana::contains(input_string,
                                       boost::hana::char_c<'-'>)) {
    auto input = boost::hana::unpack(
        input_string, [](auto... c) { return boost::hana::make_tuple(c...); });
    auto remove_comma = boost::hana::remove_if(input, [](auto c) {
      return boost::hana::bool_c<c == boost::hana::char_c<','>>;
    });

    auto result_str =
        boost::hana::concat(boost::hana::make_tuple(boost::hana::char_c<'-'>,
                                                    boost::hana::char_c<'>'>),
                            stable_unique(remove_comma));
    auto new_str = boost::hana::concat(input, result_str);
    return make_label_from_inputs(tuple_to_string(new_str));
  } else {
    auto lrout = parse_input(input_string);
    auto l = boost::hana::at_c<0>(lrout);
    auto r = boost::hana::at_c<1>(lrout);
    auto out = boost::hana::at_c<2>(lrout);
    return make_labels(std::move(l), std::move(r), std::move(out));
  }
}

consteval auto parse_unary_input(auto &&input_string) {
  auto [l, out] = boost::hana::unpack(input_string, [](auto &&...chars) {
    auto input = boost::hana::make_tuple(chars...);

    constexpr auto pos_arrow = boost::hana::index_if(
        input, [](auto &&c) { return c == boost::hana::char_c<'-'>; });
    auto ls = boost::hana::take_front(input, *pos_arrow);
    auto final =
        boost::hana::drop_front(input, *pos_arrow + boost::hana::size_c<2>);

    return std::make_tuple(tuple_to_string(ls), tuple_to_string(final));
  });
  return boost::hana::make_tuple(std::move(l), std::move(out));
}

consteval auto make_uni_label_from_inputs(auto input_string) {
  if constexpr (!boost::hana::contains(input_string,
                                       boost::hana::char_c<'-'>)) {
    auto input = boost::hana::unpack(
        input_string, [](auto... c) { return boost::hana::make_tuple(c...); });

    auto non_repeated = boost::hana::filter(stable_unique(input), [&](auto x) {
      auto count = boost::hana::fold_left(
          input, boost::hana::size_c<0>, [&](auto acc, auto c) {
            return boost::hana::if_(c == x, acc + boost::hana::size_c<1>, acc);
          });
      return count == boost::hana::size_c<1>;
    });

    auto arrow_tail =
        boost::hana::concat(boost::hana::make_tuple(boost::hana::char_c<'-'>,
                                                    boost::hana::char_c<'>'>),
                            non_repeated);
    auto new_str = boost::hana::concat(input, arrow_tail);
    return make_uni_label_from_inputs(tuple_to_string(new_str));
  } else {
    auto lout = parse_unary_input(input_string);
    auto l = boost::hana::at_c<0>(lout);
    auto out = boost::hana::at_c<1>(lout);
    return make_uni_labels(std::move(l), std::move(out));
  }
}

consteval auto parse_n_input(auto input_string) {
  auto chars = boost::hana::unpack(input_string, boost::hana::make_tuple);

  constexpr auto pos_arrow = boost::hana::index_if(
      chars, [](auto c) { return c == boost::hana::char_c<'-'>; });
  auto prefix = boost::hana::take_front(chars, *pos_arrow);
  auto output_chars =
      boost::hana::drop_front(chars, *pos_arrow + boost::hana::size_c<2>);

  auto fold_result = boost::hana::fold_left(
      prefix,
      boost::hana::make_pair(boost::hana::make_tuple(),
                             boost::hana::make_tuple()),
      [](auto acc, auto c) {
        auto segs = boost::hana::first(acc);
        auto cur = boost::hana::second(acc);
        return boost::hana::if_(
            c == boost::hana::char_c<','>,
            boost::hana::make_pair(
                boost::hana::append(segs, tuple_to_string(cur)),
                boost::hana::make_tuple()),
            boost::hana::make_pair(segs, boost::hana::append(cur, c)));
      });

  auto all_input_strings =
      boost::hana::append(boost::hana::first(fold_result),
                          tuple_to_string(boost::hana::second(fold_result)));
  auto output_string = tuple_to_string(output_chars);
  return boost::hana::make_pair(all_input_strings, output_string);
}

consteval auto make_n_label_from_inputs(auto input_string) {
  if constexpr (!boost::hana::contains(input_string,
                                       boost::hana::char_c<'-'>)) {
    auto chars = boost::hana::unpack(input_string, boost::hana::make_tuple);
    auto no_comma = boost::hana::remove_if(
        chars, [](auto c) { return c == boost::hana::char_c<','>; });
    auto non_repeated =
        boost::hana::filter(stable_unique(no_comma), [&](auto x) {
          auto count = boost::hana::fold_left(
              no_comma, boost::hana::size_c<0>, [&](auto acc, auto c) {
                return boost::hana::if_(c == x, acc + boost::hana::size_c<1>,
                                        acc);
              });
          return count == boost::hana::size_c<1>;
        });
    auto arrow_tail =
        boost::hana::concat(boost::hana::make_tuple(boost::hana::char_c<'-'>,
                                                    boost::hana::char_c<'>'>),
                            non_repeated);
    return make_n_label_from_inputs(
        tuple_to_string(boost::hana::concat(chars, arrow_tail)));
  } else {
    auto parsed = parse_n_input(input_string);
    auto input_strings = boost::hana::first(parsed);
    auto output_string = boost::hana::second(parsed);
    return boost::hana::unpack(input_strings, [&](auto... inp) {
      return make_n_labels(output_string, inp...);
    });
  }
}

} // namespace einsum_detail

#endif // EINSTEIN_SUMMATION2_HELPERS_HPP
