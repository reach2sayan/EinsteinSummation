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
#include <string>
#include <string_view>
#include <tuple>

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
  void parse_and_create_object();
  std::string unique_tables;
  std::map<std::string, std::vector<int>> dim_to_table_map;
  std::vector<int> broadcast_dims;
  std::string broadcast_list;
  std::vector<int> combinations;
  std::map<char, int> unique_dict;
  bool contains_arrow;
  bool contains_repeated_index;

  const std::string procedure;
  const Tuple pots;
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

template <typename T>
std::vector<int> make_dimension_tuple(const T& obj);

template <typename Tuple, std::size_t... Is>
void tuple_to_map_impl(std::map<std::string, std::vector<int>>& m,
		       const Tuple& t, const std::vector<std::string>& tables,
		       std::index_sequence<Is...>) {
  (..., (m[tables[Is]] = make_dimension_tuple(std::get<Is>(t))));
}

template <typename... Args>
std::map<std::string, std::vector<int>> tupleToMap(
    const std::tuple<Args...>& pots, const std::vector<std::string>& tables) {
  std::map<std::string, std::vector<int>> m;
  tuple_to_map_impl(m, pots, tables, std::index_sequence_for<Args...>{});
  return m;
}

std::vector<std::string> string_split(const std::string& str, char delimiter);
#endif
