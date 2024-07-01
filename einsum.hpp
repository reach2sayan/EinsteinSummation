#if !defined(EINSTEIN_SUMMATION_HPP__)
#define EINSTEIN_SUMMATION_HPP__

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <regex>
#include <set>
#include <string_view>
#include <tuple>

#include "expression.hpp"
#include "generator.hpp"

using Eigen::MatrixXd;

template <typename Tuple>
class EinsteinSummation {
 public:
  EinsteinSummation() = default;

  explicit EinsteinSummation(std::string&& _proc, const Tuple& matrices);
  explicit EinsteinSummation(const std::string& _proc, const Tuple& matrices);

  template <typename T, typename... TArgs>
  EinsteinSummation(T&& _proc, TArgs... matrices) {
    EinsteinSummation(std::forward<T>(_proc),
		      std::forward_as_tuple(std::forward<TArgs>(matrices)...));
  }

 private:
  Expression procedure;
  const Tuple pots;
  bool validate_procedure();
  void parse_and_create_object();
  std::string unique_tables;
  std::map<std::string, std::vector<int>> dim_to_table_map;
  std::generator<const std::vector<int>&> generate_combinations(
      const std::vector<int>& n);
  std::vector<int> broadcast_dims;
  std::string broadcast_list;
  std::vector<int> combinations;
  std::map<char, int> unique_dict;
  bool contains_arrow;
  bool contains_repeated_index;
};

template <typename Tuple>
EinsteinSummation<Tuple>::EinsteinSummation(const std::string& _proc,
					    const Tuple& matrices)
    : procedure{_proc}, pots{matrices} {
  parse_and_create_object();
}
template <typename Tuple>
EinsteinSummation<Tuple>::EinsteinSummation(std::string&& _proc,
					    const Tuple& matrices)
    : procedure{_proc}, pots{matrices} {
  parse_and_create_object();
}

inline auto diagonal(const MatrixXd& pot) { return pot.diagonal(); }
inline auto diagonal_sum(const MatrixXd& pot) { return pot.diagonal().sum(); }

#endif

#if defined(SHOW_OLD_CODE)
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
		 return (it != this->unique_dict.end()) ? it->second
							: KEY_NOT_FOUND_DEFAULT;
	       });

combinations.erase(std::remove(combinations.begin(), combinations.end(),
			       KEY_NOT_FOUND_DEFAULT),
		   combinations.end());
#endif

