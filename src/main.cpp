#include "NeuralNetwork.hpp"
#include <chrono>
#include <iostream>

int main() {
    // std::vector<size_t> n = {1600, 800, 400, 4};
    std::vector<size_t> n = {64, 2000, 2000, 3000, 3000};
    NeuralNetwork network = createNewNetwork(n);

    const int runs = 100;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> inputLayer(64, 0.5f);
    std::span<float> out; // = processData(inputLayer, network);

    for (int i = 0; i < runs; i++) {
        out = processData(inputLayer, network);
        mutateNet(network);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg = static_cast<double>(total.count()) / runs;

    std::cout << "Total: " << total.count() << " ms para " << runs << " runs" << std::endl;
    std::cout << "Promedio: " << avg << " ms por run" << std::endl;

    // for (int i = 0; i < runs; i++) {
    //     std::span<float> out = processData(inputLayer, network);
    //     mutateNet(network);
    // }

    return 0;
}