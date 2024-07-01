#include "einsum.hpp"

std::generator<const std::vector<int>&> combo(const std::vector<int>& n) {
  std::vector<int> r(n.size(), 0);
  while (true) {
    co_yield r;
    int p = r.size() - 1;
    r[p] += 1;
    while (r[p] == n[p]) {
      r[p] = 0;
      p -= 1;
      if (p < 0)
	co_return;
      else
	r[p] += 1;
    }
  }
}
template <typename Tuple>
std::generator<const std::vector<int>&>
EinsteinSummation<Tuple>::generate_combinations(const std::vector<int>& n) {
  std::vector<int> r(n.size(), 0);
  while (true) {
    co_yield r;
    int p = r.size() - 1;
    r[p] += 1;
    while (r[p] == n[p]) {
      r[p] = 0;
      p -= 1;
      if (p < 0)
	co_return;
      else
	r[p] += 1;
    }
  }
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
  const auto& left_expression = procedure.get_left_expression();
  auto dim_to_table_map = tupleToMap(pots, left_expression.get_tables());

  bool left_expression_is_valid =
      left_expression.validate_expression(dim_to_table_map);
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

  int l4 = 0;
  for (const std::vector<int>& x : combo(combinations)) {
    for (int i : x) std::cout << i << ",";
    std::cout << std::endl;
    ++l4;
  }
  return 0;
}
