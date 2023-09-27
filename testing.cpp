#include <iostream>
#include <vector>

template <typename T> void printVector(const std::vector<T> &vec) {
  for (const auto &item : vec) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

int main() {
  std::vector<int> values = {1, 2, 3, 4, 5};
  printVector(values);

  std::vector<std::string> strValues = {"a", "b", "c"};
  printVector(strValues);

  return 0;
}
