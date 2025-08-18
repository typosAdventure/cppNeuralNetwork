#include <iostream>
#include "NeuralNetwork.hpp"
#include <chrono>

int main() {
    std::vector<size_t> n = {64, 2000, 2000, 3000, 3000};
    NeuralNetwork network = createNewNetwork(n);

    std::vector<float> inputLayer(64, 0.5f);

    const int runs = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; i++) {
        auto out = processData(inputLayer, network);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg = static_cast<double>(total.count()) / runs;

    std::cout << "Total: " << total.count() << " ms para " << runs << " runs" << std::endl;
    std::cout << "Promedio: " << avg << " ms por run" << std::endl;

    return 0;
}