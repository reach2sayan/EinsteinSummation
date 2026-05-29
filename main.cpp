#include "einsum_single.hpp"

#include <vector>

template <typename... T> struct TD;
int main() {
  std::vector A2{11, 12, 13, 14, 21, 22, 23, 24,
                 31, 32, 33, 34, 41, 42, 43, 44};
  std::vector B2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

  std::mdspan<int, std::extents<std::size_t, 4, 4>> mdA2{A2.data()};
  std::mdspan<int, std::extents<std::size_t, 4, 4>> mdB2{B2.data()};
  std::vector A{1, 2, 3, 4};
  std::vector<int> B{1,2,3,4};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{B.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{A.data()};
  make_einsum(einsum, "ij,ij", mdA, mdB);
  einsum.eval();
  const auto &res_data = einsum.get_result();

  for (std::size_t i = 0; i < 2; i++) {
    for (std::size_t j = 0; j < 2; j++) {
      std::cout << res_data[i * 2 + j] << " ";
    }
    std::cout << "\n";
  }
}
