#pragma once
#include <algorithm>
#include <random>
#include <span>
#include <vector>

struct NeuralNetwork {
    std::vector<float> values;
    std::vector<std::span<float>> spanValues;
    std::vector<float> weights;
    std::vector<std::span<float>> spanWeights;
    std::vector<size_t> inputsPerNode;
};

float randomWeight();
float activationFunction(float number);
void sumNextLayer(std::span<float> currVs, std::span<float> currWs, std::span<float> nextVs);

// Creates a new instance of a neuralNetwork with random float values between
// -0.05 and +0.05 for the weights (to be modified in the future).
// Parameter: vector.size() represents the amount of layers, each value
//     determines the nodes per layer.
NeuralNetwork createNewNetwork(std::vector<size_t> nodesPerLayer);
std::span<float> processData(std::vector<float> &input, NeuralNetwork &net);
void mutateNet(NeuralNetwork &net);