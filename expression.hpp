#if !defined __EXPRESSION_HPP__
#define __EXPRESSION_HPP__

#include <map>
#include <string>
#include <vector>

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

class Expression {
 private:
  std::string expr;
  class LeftExpression {
   public:
    LeftExpression() = default;
    LeftExpression(std::string&& s)
	: lexpr{s}, tables{string_split(lexpr, ',')} {};
    std::vector<std::string> get_tables() const { return tables; }

    template <typename Tuple>
    bool validate_expression(const Tuple& pots) const;
    const std::string& get_expression_view() const { return lexpr; }

   private:
    std::string lexpr;
    std::vector<std::string> tables;
  };

  class RightExpression {
   public:
    RightExpression() = default;
    RightExpression(std::string&& s) : rexpr{s} {}

    bool validate_expression(const LeftExpression& l) const;
    const std::string& get_expression_view() const { return rexpr; }

   private:
    std::string rexpr;
  };

  LeftExpression LExpr;
  RightExpression RExpr;

 public:
  Expression(const std::string&);
  const LeftExpression& get_left_expression() const { return LExpr; }
};

#endif
