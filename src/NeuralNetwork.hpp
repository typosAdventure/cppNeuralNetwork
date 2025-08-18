#pragma once
#include <vector>
#include <random>
#include <algorithm>

struct Layer {
    std::vector<float> values;
    std::vector<float> weights;
    size_t inputsPerNode;
};

struct NeuralNetwork {
    std::vector<Layer> layers;
};

float randomWeight();
float activationFunction(float number);

NeuralNetwork createNewNetwork(std::vector<size_t> nodesPerLayer);
std::vector<float> processData(const std::vector<float>& input, NeuralNetwork& net);