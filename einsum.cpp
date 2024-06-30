#include "einsum.hpp"

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

auto parse_string(const std::string& procedure)
    -> std::tuple<std::vector<std::string>, std::string> {
  std::vector<std::string> halves = string_split(procedure, '/');
  std::vector<std::string> tables = string_split(halves.front(), ',');
  std::string broadcast = halves[1];

  return {std::move(tables), std::move(broadcast)};
}

template <typename T>
auto make_dimension_tuple(const T& obj) -> std::vector<int> {
  std::vector<int> retlist;
  retlist.reserve(obj.NumDimensions);
  std::ranges::iota_view<int, int> indices{0, obj.NumDimensions};
  std::transform(indices.begin(), indices.end(), std::back_inserter(retlist),
		 [&](int i) { return obj.dimensions()[i]; });
  return retlist;
}

std::string remove_duplicates(const std::string& input) {
  std::string result;
  std::set<char> seen;

  auto found_unique_char = [&seen](char c) -> bool {
    if (seen.find(c) == seen.end()) {
      seen.insert(c);
      return true;
    } else
      return false;
  };
  std::copy_if(input.begin(), input.end(), std::back_inserter(result),
	       std::move(found_unique_char));
  return result;
}

template <typename Tuple>
void EinsteinSummation<Tuple>::parse_and_create_object() {
  std::regex regexp{"(.)\1"};
  contains_repeated_index = std::regex_search(procedure, regexp);
  contains_arrow = true;  // procedure.contains("->"sv);

  // if (containsArrow && match) return std::get<0>(pots).diagonal();
  // if (containsArrow) return std::get<0>(pots).diagonal().sum();
  //
  auto [tabs, bds] = parse_string(procedure);

  auto dim_to_table_map = tupleToMap(pots, tabs);

  std::string flat_tables =
      std::accumulate(tabs.begin(), tabs.end(), std::string(""));
  unique_tables = std::ranges::sort(remove_duplicates(flat_tables));

  std::vector<int> flat_dim;
  for (const auto& [key, vec] : dim_to_table_map) {
    flat_dim.insert(flat_dim.end(), vec.begin(), vec.end());
  }

  for (const auto& [index, letter] : std::views::enumerate(flat_tables))
    unique_dict[letter] = flat_dim[index];

  const int KEY_NOT_FOUND_DEFAULT = -1;
  std::vector<int> combinations;
  std::transform(unique_tables.begin(), unique_tables.end(),
		 std::back_inserter(combinations), [this](char z) -> int {
		   auto it = this->unique_dict.find(z);
		   return (it != this->unique_dict.end())
			      ? it->second
			      : KEY_NOT_FOUND_DEFAULT;
		 });

  combinations.erase(std::remove(combinations.begin(), combinations.end(),
				 KEY_NOT_FOUND_DEFAULT),
		     combinations.end());
}

int main() {
  MatrixXd m(5, 5);
  m << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24;
  std::cout << m << std::endl;
  std::cout << diagonal(m).transpose() << std::endl;

  std::vector<double> l(60);
  std::iota(l.begin(), l.end(), 0);
  Eigen::TensorFixedSize<double, Eigen::Sizes<3, 4, 5>> a;
  a = Eigen::TensorMap<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 4, 5>>>(
      l.data(), 3, 4, 5);

  std::vector<double> l2(24);
  std::iota(l2.begin(), l2.end(), 0);
  Eigen::TensorFixedSize<double, Eigen::Sizes<4, 3, 2>> b;
  b = Eigen::TensorMap<Eigen::TensorFixedSize<double, Eigen::Sizes<4, 3, 2>>>(
      l.data(), 4, 3, 2);

  std::string procedure{"ijk,jil/kl"};
  auto [tabs, bds] = parse_string(procedure);
  std::cout << "Tables\n";
  for (auto table : tabs) std::cout << table << std::endl;
  std::cout << "Broadcast\n";
  std::cout << bds << std::endl;

  auto dim_as_t = make_dimension_tuple(a);
  auto mymap = tupleToMap(std::make_tuple(a, b), tabs);

  for (auto&& [key, val] : mymap) {
    std::cout << "Key " << key << std::endl;
    for (auto&& v : val) std::cout << v << ", ";
    std::cout << std::endl;
  }

  std::string flat_tables =
      std::accumulate(tabs.begin(), tabs.end(), std::string(""));
  std::cout << "Flat Tables " << flat_tables << std::endl;
  std::string original_tables = remove_duplicates(flat_tables);
  std::cout << "Original Tables " << original_tables << std::endl;
  std::ranges::sort(original_tables);

  std::string unique_tables = original_tables;
  std::vector<int> flat_dim;
  for (const auto& [key, vec] : mymap) {
    flat_dim.insert(flat_dim.end(), vec.begin(), vec.end());
  }
  std::cout << "Flat Dims\n";
  for (auto&& v : flat_dim) std::cout << v << ", ";
  std::cout << std::endl;

  std::map<char, int> unique_dict;
  for (const auto& [index, letter] : std::views::enumerate(flat_tables))
    unique_dict[letter] = flat_dim[index];

  const int KEY_NOT_FOUND_DEFAULT = -1;
  std::vector<int> combinations;
  std::transform(
      unique_tables.begin(), unique_tables.end(),
      std::back_inserter(combinations), [&unique_dict](char z) -> int {
	auto it = unique_dict.find(z);
	return (it != unique_dict.end()) ? it->second : KEY_NOT_FOUND_DEFAULT;
      });

  combinations.erase(std::remove(combinations.begin(), combinations.end(),
				 KEY_NOT_FOUND_DEFAULT),
		     combinations.end());
  std::cout << "Combinations\n";
  for (auto&& v : combinations) std::cout << v << ", ";

  return 0;
}
