#include "NeuralNetwork.hpp"
#include <iostream>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dist(
    std::nextafter(-0.5f, 0.0f), 
    std::nextafter(0.5f, 0.0f)
);
static std::uniform_real_distribution<float> mutation(
    std::nextafter(-0.05f, 0.0f), 
    std::nextafter(0.05f, 0.0f)
);

float randomMutation() {
    // float x = mutation(gen);

    return mutation(gen);
}

float randomWeight() {
    float x = dist(gen);

    return x;
}

float activationFunction(float number) {
    float scope = 4.0f;
    float res = 0;

    if (number >= 0) {
        res = (-scope / (scope + number)) + 1;
    } else {
        res = -((-scope / (scope - number)) + 1);
    }

    return (std::fabs(res) < 1e-6f) ? 0.0f : res;
}

// Creates a new instance of a neuralNetwork with random float values between
// -1 and +1 for the weights (to be modified in the future).
// Parameter: vector.size() represents the amount of layers, each value
//     determines the nodes per layer.
NeuralNetwork createNewNetwork(std::vector<size_t> nodesPerLayer) {
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

// Returns the sum of the current value
float sumValuesCurr(const Layer& curr, size_t offSet) {
    auto& vals = curr.values;
    auto& weights = curr.weights;
    float sum = 0.0f;
    
    for (size_t i = 0; i < vals.size(); i++) {
        sum += vals[i] * weights[i + offSet];
    }

    return activationFunction(sum);
}

// For each node in the next layer, saves de value of curr.value * curr.weight
void sumValuesNextLayer(const Layer& curr, std::vector<float>& nextValues) {
    for (size_t i = 0; i < nextValues.size(); i++) {
        nextValues[i] = sumValuesCurr(curr, curr.inputsPerNode * i);
    }
}

// Recieves the input layer, processes the data and returns last layer as an output.
std::vector<float> processData(const std::vector<float>& inputLayer, NeuralNetwork& net) {
    net.layers[0].values = inputLayer;
    
    for (size_t i = 0; i < net.layers.size() - 1; i++) {
        auto& curr = net.layers[i];
        auto& nextValues = net.layers[i + 1].values;

        sumValuesNextLayer(curr, nextValues);
    }

    return net.layers.back().values;
}

float maxMinWeight(float weight) {
    if (weight <= 0) {
        return std::min(weight, -0.9999f);
    } else {
        return std::max(weight, 0.9999f);
    }
}

void mutateWeights(std::vector<float>& weights) {
    std::vector<float> weightsMut (weights.size());
    std::generate(weightsMut.begin(), weightsMut.end(), randomMutation);

    for (size_t i = 0; i < weights.size(); i++) {
        // weights[i] = maxMinWeight(weights[i] + mutation(gen));
        weights[i] = maxMinWeight(weights[i] + weightsMut[i]);
    }
}

void mutateNet(NeuralNetwork& net) {
    for (size_t i = 0; i < net.layers.size(); i++) {
        auto& curr = net.layers[i];
        
        mutateWeights(curr.weights);
    }
}