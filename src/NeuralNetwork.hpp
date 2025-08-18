#pragma once
#include <vector>
#include <random>
#include <algorithm>

// std::vector<float> values;      // todas las activaciones de todos los nodos
// std::vector<float> weights;     // todos los pesos
// std::vector<int> layer_sizes;   // cantidad de nodos por capa
// std::vector<int> layer_offsets; // índices de arranque de cada capa en "values"

struct Layer {
    std::vector<float> values;
    std::vector<float> weights;
    int inputsPerNode;
};

struct NeuralNetwork {
    std::vector<Layer> layers;
};

NeuralNetwork createNewNetwork(std::vector<int> nodesPerLayer);

float randomWeight();