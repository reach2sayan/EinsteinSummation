#include "expression.hpp"

#include <cassert>
#include <ranges>
#include <tuple>

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
