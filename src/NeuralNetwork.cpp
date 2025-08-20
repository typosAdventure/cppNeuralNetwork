#include "NeuralNetwork.hpp"
#include <iostream>
#include <numeric>

// Initializin random seeds
static std::random_device rd;
static std::mt19937 gen(rd());

// Weights
static std::uniform_real_distribution<float> dist(
    std::nextafter(-0.5f, 0.0f),
    std::nextafter(0.5f, 0.0f));

// Mutations
static std::uniform_real_distribution<float> mutation(
    std::nextafter(-0.05f, 0.0f),
    std::nextafter(0.05f, 0.0f));

float randomMutation() {
    return mutation(gen);
}

inline float randomWeight() {
    return dist(gen);
}

inline float activationFunction(float number) {
    float scope = 4.0f;
    float res = 0;

    if (number >= 0) {
        res = (-scope / (scope + number)) + 1;
    } else {
        res = -((-scope / (scope + number)) + 1);
    }

    return (std::abs(res) < 1e-6f) ? 0.0f : res;
}

std::vector<std::span<float>> makeSpans(std::vector<float> &buffer, std::span<const size_t> sizes) {
    size_t need = 0;
    for (size_t s : sizes) {
        need += s;
    }

    if (need > buffer.size()) {
        throw std::runtime_error("Sizes greater than buffer");
    }

    std::vector<std::span<float>> out;
    out.reserve(sizes.size());

    size_t offset = 0;
    for (size_t s : sizes) {
        out.emplace_back(buffer.data() + offset, s);
        offset += s;
    }

    return out;
}

// Returns the sum of the current value.
// #pragma GCC ivdep;
inline float sumValuesCurr(std::span<float> currVs, std::span<float> currWs) {
    float sum = 0.0f;

    for (size_t i = 0; i < currVs.size(); i++) {
        sum += currVs[i] * currWs[i];
    }

    return activationFunction(sum);
}

// Computes the value of each node in the next layer.
void sumNextLayer(std::span<float> currVs, std::span<float> currWs, std::span<float> nextVs) {
    auto stride = currVs.size();
    for (size_t i = 0; i < nextVs.size(); i++) {
        auto ws = currWs.subspan(i * stride, stride);
        nextVs[i] = sumValuesCurr(currVs, ws);
    }
}

float maxMinWeight(float weight) {
    return std::clamp(weight, -0.9999f, 0.9999f);
}

// Creates a new instance of a neuralNetwork with random float values between
// -1 and +1 for the weights (to be modified in the future)
// Parameter: vector.size() represents the amount of layers, each value
//     determines the nodes per layer.
NeuralNetwork createNewNetwork(std::vector<size_t> nodesPerLayer) {
    NeuralNetwork net;

    size_t sumValues = std::accumulate(nodesPerLayer.begin(), nodesPerLayer.end(), 0);
    size_t sumWeights = 0;

    for (size_t i = 0; i < nodesPerLayer.size() - 1; i++) {
        sumWeights += nodesPerLayer[i] * nodesPerLayer[i + 1];
    }

    net.inputsPerNode = nodesPerLayer;
    net.values = std::vector<float>(sumValues, 0.0f);
    net.weights = std::vector<float>(sumWeights);
    std::generate(net.weights.begin(), net.weights.end(), randomWeight);

    net.spanValues = makeSpans(net.values, net.inputsPerNode);
    net.spanWeights = makeSpans(net.weights, net.inputsPerNode);

    return net;
}

// Recieves the input layer, processes the data and returns last layer as an output.
std::span<float> processData(std::vector<float> &input, NeuralNetwork &net) {
    size_t layers = net.inputsPerNode.size();

    for (size_t i = 0; i < input.size(); i++) {
        net.values[i] = input[i];
    }

    for (size_t i = 0; i < layers - 1; i++) {
        auto &currValues = net.spanValues[i];
        auto &currWeights = net.spanWeights[i];
        auto &nextValues = net.spanValues[i + 1];

        sumNextLayer(currValues, currWeights, nextValues);
    }

    return net.spanValues[layers - 1];
}

// Mutates every weight of the net.
void mutateNet(NeuralNetwork &net) {
    auto &weights = net.weights;

    std::vector<float> mutations;
    mutations.reserve(weights.size());
    std::generate(mutations.begin(), mutations.end(), randomMutation);

    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = maxMinWeight(weights[i] + mutations[i]);
    }
}