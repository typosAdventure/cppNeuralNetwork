#include "NeuralNetwork.hpp"

float randomWeight() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(
        std::nextafter(-1.0f, 0.0f), 
        std::nextafter(1.0f, 0.0f)
    );

    float x = dist(gen);

    return x;
}

NeuralNetwork createNewNetwork(std::vector<int> nodesPerLayer) {
    NeuralNetwork neuralNetwork;
    
    for (size_t i = 0; i < nodesPerLayer.size(); i++) {
        Layer layer;
        layer.values = std::vector<float> (nodesPerLayer[i], 0.0f);

        if (i + 1 < nodesPerLayer.size()) {
            layer.weights = std::vector<float>(nodesPerLayer[i] * nodesPerLayer[i + 1]);
            std::generate(layer.weights.begin(), layer.weights.end(), randomWeight);
            
            layer.inputsPerNode = nodesPerLayer[i];
        } else {
            layer.weights = std::vector<float>(0);
            layer.inputsPerNode = 0;
        }
        
        neuralNetwork.layers.push_back(layer);
    }

    return neuralNetwork;
}