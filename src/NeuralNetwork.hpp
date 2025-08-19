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

struct NeuralNetworkBIS {
    std::vector<float> values;
    std::vector<float> weights;
    std::vector<size_t> inputsPerNode;
};

float randomWeight();
float activationFunction(float number);

// Creates a new instance of a neuralNetwork with random float values between
// -0.05 and +0.05 for the weights (to be modified in the future).
// Parameter: vector.size() represents the amount of layers, each value
//     determines the nodes per layer.
NeuralNetwork createNewNetwork(std::vector<size_t> nodesPerLayer);
std::vector<float> processData(const std::vector<float>& input, NeuralNetwork& net);
void mutateNet(NeuralNetwork& net);