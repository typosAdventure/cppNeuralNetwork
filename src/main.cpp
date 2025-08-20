#include "NeuralNetwork.hpp"
#include <chrono>
#include <iostream>

int main() {
    // std::vector<size_t> n = {1600, 800, 400, 4};;
    std::vector<size_t> n = {700, 1400, 2500, 3600};
    NeuralNetwork network = createNewNetwork(n);

    const int runs = 1000;

    std::vector<float> inputLayer(64, 0.5f);
    std::span<float> out; // = processData(inputLayer, network);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; i++) {
        out = processData(inputLayer, network);
        mutateNet(network);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg = static_cast<double>(total.count()) / runs;

    std::cout << "Total: " << total.count() << " ms para " << runs << " runs" << std::endl;
    std::cout << "Promedio: " << avg << " ms por run" << std::endl;

    return 0;
}