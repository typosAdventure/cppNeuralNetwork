#include "NeuralNetwork.hpp"
#include <chrono>
#include <iostream>

int main() {
    int sum;    

    std::vector<size_t> n = {64, 1500, 200, 4};
    NeuralNetwork network = createNewNetwork(n);

    const int runs = 10;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> inputLayer(64, 0.5f);

    for (int i = 0; i < runs; i++) {
        auto out = processData(inputLayer, network);
        mutateNet(network);
    }

    // auto out = processData(inputLayer, network);

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg = static_cast<double>(total.count()) / runs;

    std::cout << "Total: " << total.count() << " ms para " << runs << " runs" << std::endl;
    std::cout << "Promedio: " << avg << " ms por run" << std::endl;

    return 0;
}