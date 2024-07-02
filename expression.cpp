#include "expression.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <ranges>
#include <set>

template <typename T>
auto make_dimension_tuple(const T& obj) -> std::vector<int> {
  std::vector<int> retlist;
  retlist.reserve(obj.NumDimensions);
  std::ranges::iota_view<int, int> indices{0, obj.NumDimensions};
  std::transform(indices.begin(), indices.end(), std::back_inserter(retlist),
		 [&](int i) { return obj.dimensions()[i]; });
  return retlist;
}

auto string_split(const std::string& str, char delimiter)
    -> std::vector<std::string> {
  auto to_string = [](auto&& r) -> std::string {
    auto data = &(*r.begin());
    size_t size = static_cast<std::size_t>(std::ranges::distance(r));
    return {data, size};
  };

  auto range = str | std::ranges::views::split(delimiter) |
	       std::ranges::views::transform(to_string);

  return {range.begin(), range.end()};
}

Expression::Expression(const std::string& procedure) {
  expr = procedure;
  std::vector<std::string> halves = string_split(procedure, '/');
  assert(halves.size() == 2);
  RExpr = std::move(halves.front());
  LExpr = std::move(*halves.rbegin());
}

bool has_unique_characters(
    const std::pair<std::string, std::vector<int>>& str) {
  std::set<char> seenCharacters;

  for (char c : str.first) {
    auto result = seenCharacters.insert(c);
    if (!result.second) return false;
  }
  return true;
}

template <typename Tuple>
bool Expression::LeftExpression::validate_expression(const Tuple& pots) const {
  std::map<std::string, std::vector<int>> indices_and_dims =
      tupleToMap(pots, tables);

  constexpr auto all_keys_have_unique_chars = [&indices_and_dims]() -> bool {
    return std::ranges::all_of(indices_and_dims, has_unique_characters);
  };

  constexpr auto key_and_indices_are_equal_length =
      [&indices_and_dims]() -> bool {
    return std::ranges::all_of(indices_and_dims, [](const auto& pair) {
      return pair.first.length() == pair.second.size();
    });
  };

  constexpr auto equal_axis_has_equal_dimensions = [&indices_and_dims]() {
    std::map<char, int> split_indices_and_dims_map;
    for (const auto& [indices, dims] : split_indices_and_dims_map) {
      for (const auto& indices_and_dims :
	   std::ranges::views::zip(indices, dims)) {
	if (split_indices_and_dims_map.find(std::get<0>(indices_and_dims)) ==
	    split_indices_and_dims_map.end())
	  split_indices_and_dims_map[std::get<0>(indices_and_dims)] =
	      std::get<1>(indices_and_dims);
	else if (split_indices_and_dims_map.at(std::get<0>(indices_and_dims)) !=
		 std::get<1>(indices_and_dims))
	  return false;
      }
    }
  };

  if (equal_axis_has_equal_dimensions() and
      key_and_indices_are_equal_length() and all_keys_have_unique_chars())
    return true;
  else
    return false;
}

bool Expression::RightExpression::validate_expression(
    const LeftExpression& l) const {
  auto extract_non_duplicated_chars =
      [](std::string&& combined_lhe) -> std::string {
    std::map<char, int> charCount;
    for (char c : combined_lhe) charCount[c]++;
    auto uniqueChars =
	combined_lhe | std::ranges::views::filter(
			   [&charCount](char c) { return charCount[c] == 1; });

    return std::string{uniqueChars.begin(), uniqueChars.end()};
  };

  std::string combined_left_hand_expression = std::accumulate(
      l.get_tables().begin(), l.get_tables().end(), std::string{});

  return rexpr ==
	 extract_non_duplicated_chars(std::move(combined_left_hand_expression));
}
