#include <iostream>
#include "NeuralNetwork.hpp"

int main() {
    std::vector<int> n = {4,10,100,45000, 10};
    NeuralNetwork network = createNewNetwork(n);

    
    for (size_t i = 0; i < n.size(); i++) {
        std::cout << network.layers[i].inputsPerNode << std::endl;
    }
    
    // std::cout << network.layers.size() << std::endl;

    return 0;
}