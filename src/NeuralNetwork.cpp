#include "nn/NeuralNetwork.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <omp.h>

// Initializing random seeds
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

float randomWeight() {
    return dist(gen);
}

float activationFunction(float number) {
    float scope = 1.0f;
    float res = 0;

    if (number >= 0) {
        res = (-scope / (scope + number)) + 1;
    } else {
        res = -((-scope / (scope - number)) + 1);
    }

    return (std::abs(res) < 1e-6f) ? 0.0f : res;

    // if (number > 1) {
    //     number = 1;
    // } else if (number < -1) {
    //     number = -1;
    // }

    // return number;
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

// Computes the value of each node in the next layer.
void sumNextLayer(std::span<float> currVs, std::span<float> currWs, std::span<float> nextVs) {
    for (size_t i = 0; i < nextVs.size(); i++) {
        nextVs[i] = 0;
    }

    // #pragma omp parallel for simd
    for (size_t i = 0; i < currVs.size(); ++i) { // outer
        // float sum = 0;
        float vsi = currVs[i]; // se queda en registro
        // float sum = 0.0f;
        size_t offset = i * nextVs.size();

        for (size_t j = 0; j < nextVs.size(); ++j) {
            nextVs[j] += currWs[offset + j] * vsi;
        }
        // nextVs[j] = sum;
    }

    for (auto &val : nextVs) {
        val = activationFunction(val);
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

    size_t sumValues = std::accumulate(nodesPerLayer.begin(), nodesPerLayer.end(), size_t{0});
    size_t sumWeights = 0;
    // std::vector<size_t> weightsPerLayer;

    for (size_t i = 0; i + 1 < nodesPerLayer.size(); ++i) {
        sumWeights += nodesPerLayer[i] * nodesPerLayer[i + 1];
        net.weightsPerLayer.push_back(nodesPerLayer[i] * nodesPerLayer[i + 1]);
    }

    net.inputsPerNode = nodesPerLayer;
    net.values = std::vector<float>(sumValues, 0.0f);
    net.weights = std::vector<float>(sumWeights);
    std::generate(net.weights.begin(), net.weights.end(), randomWeight);

    net.spanValues = makeSpans(net.values, net.inputsPerNode);
    net.spanWeights = makeSpans(net.weights, net.weightsPerLayer);

    return net;
}

// Recieves the input layer, processes the data and returns last layer as an output.
std::span<float> processData(std::vector<float> &input, NeuralNetwork &net) {
    net.spanValues = makeSpans(net.values, net.inputsPerNode);
    net.spanWeights = makeSpans(net.weights, net.weightsPerLayer);

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

    // #pragma omp parallel for
    for (size_t i = 0; i < weights.size(); i++) {
        static thread_local uint32_t seed = 123456789u ^ omp_get_thread_num();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float mutation = -0.05f + 0.1f * (seed / static_cast<float>(UINT32_MAX));
        weights[i] = std::clamp(weights[i] + mutation, -0.9999f, 0.9999f);
    }
}

// Mutates every weight of the net.
NeuralNetwork mutateNetBIS(NeuralNetwork &net, std::vector<size_t> n, float percentage) {
    NeuralNetwork newNet = createNewNetwork(n);

    newNet.weights = net.weights;
    auto &weights = newNet.weights;

    // #pragma omp parallel for
    for (size_t i = 0; i < weights.size(); i++) {
        float n = percentage / 100;
        static thread_local uint32_t seed = 123456789u ^ omp_get_thread_num();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float mutation = -n + ((2 * n) * (seed / static_cast<float>(UINT32_MAX)));
        weights[i] = std::clamp(weights[i] + mutation, -0.9999f, 0.9999f);
    }

    // newNet.spanValues = makeSpans(newNet.values, newNet.inputsPerNode);
    // newNet.spanWeights = makeSpans(newNet.weights, newNet.inputsPerNode);

    return newNet;
}