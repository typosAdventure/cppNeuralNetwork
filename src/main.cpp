#include "nn/NeuralNetwork.hpp"
#include <chrono>
#include <iostream>

// std::vector<float> tempWeights = {-0.0238926, -0.372677, -0.475606, 0.400088, 0.124722, 0.43801, -0.178355, -0.0539706, -0.0551936, -0.149174, -0.129072, -0.397259, -0.277207, 0.177678, 0.174177, -0.474858, -0.226324, 0.00987191, -0.20031, 0.0647259, 0.131654, 0.331396, 0.00197773, 0.188661, -0.314682, -0.036854, 0.00165393, 0.131103, -0.2804, 0.245696, 0.298617, -0.182229, 0.357472, 0.0540189, -0.0256757, -0.164826, 0.32911, -0.293698, -0.393224, 0.503779, -0.0704711, -0.248562, -0.504333, -0.125006, -0.140401, -0.351853, -0.454982, -0.2895, -0.331671, -0.0332583, 0.085094, 0.0775759, -0.246922, -0.285079, -0.361727, 0.220197, -0.318218, 0.0524564, -0.147998, -0.355797, -0.0436106, -0.167238, -0.305908, -0.0573561, 0.194227, 0.0936235, 0.0102114, 0.242492, -0.0164542, 0.111692, -0.13614, 0.0594893};

void printResults(std::span<float> out) {
    for (size_t j = 0; j < out.size(); j++) {
        std::cout << out[j] << ", ";
    }
    std::cout << std::endl;
}

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<int> dist(0, 2);

std::vector<float> getCorrectV(std::vector<float> colors) {
    std::vector<float> correct;

    for (size_t i = 0; i < colors.size(); i++) {
        if (colors[i] == 1) {
            correct[i] = 0;
        } else {
            correct[i] = 1;
        }
    }

    return correct;
}

int processAnswer(std::span<float> answer) {
    auto it = std::max_element(answer.begin(), answer.end());
    size_t index = std::distance(answer.begin(), it);

    return index;
}

size_t getCorrect(std::vector<float> v) {
    size_t correct = 0;

    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == 0) {
            correct = i;
        }
    }

    return correct;
}

// std::vector<float> tempWeights = {0.0616167, -0.855075, -0.913954, 0.9999, 0.9999, -0.9999, 0.619936, 0.9999, 0.526277};
float played = 10;

float playGameNN(NeuralNetwork &net) {
    std::vector<float> colors;
    size_t correctIndex = 0;
    size_t currCorrects = 0;
    float score = 0;
    size_t answer = 0;
    float gamesPlayed = 1.0f;
    int rnd;
    // net.weights = {0, 1, 1, 1, 0, 1, 1, 1, 0};

    while (gamesPlayed <= played) {
        currCorrects = 0;
        correctIndex = 0;
        answer = 0;

        while (answer == correctIndex && currCorrects < 100) {
            colors = {1, 1, 1};
            rnd = dist(gen);

            // std::cout << rnd << std::endl;
            colors[rnd] = 0;

            correctIndex = getCorrect(colors);
            // std::cout << "Correct: " << correctIndex << std::endl;
            answer = processAnswer(processData(colors, net));
            // std::cout << "Answer: " << answer << std::endl;

            if (answer == correctIndex) {
                currCorrects++;
                score++;
            }
        }

        // if (currCorrects >= 50) {
        //     std::cout << "OMG!" << std::endl;
        // }

        gamesPlayed++;
    }

    return (float)score / (gamesPlayed - 1);
}

int main() {
    size_t gens = 0;
    size_t genPrinter = 0;
    size_t mutations = 30;
    float globalScore = 0;

    std::vector<size_t> n = {3, 200, 9, 3};
    NeuralNetwork winnerNet = createNewNetwork(n);
    // winnerNet.weights = {0, 1, 1, 1, 0, 1, 1, 1, 0};

    float bestScore = 0.0f;

    while (gens < 3000) {
        float currScore = 0;
        std::vector<NeuralNetwork> nets;

        nets.push_back(winnerNet);

        for (size_t i = 1; i < mutations; i++) {
            float percentage = 100.0f / (5.0f - (100.0f / ((float)gens / 100.0f) + 2.0f));

            nets.push_back(mutateNetBIS(winnerNet, n, percentage));
        }

        size_t index = 0;
        while (index < nets.size()) {
            currScore = playGameNN(nets[index]);
            if (currScore > bestScore) {
                winnerNet = nets[index];
                bestScore = currScore;
            }

            index++;
        }
        if (currScore > globalScore) {
            globalScore = currScore;
        }

        gens++;
        genPrinter++;

        if (genPrinter == 100) {
            std::cout << "Gen: " << gens << ". Best score: " << globalScore; // << std::endl;
            std::cout << ". Winner net score: " << playGameNN(winnerNet) << std::endl;
            genPrinter = 0;
        }
    }

    std::cout << "Final score: " << globalScore << std::endl;
    std::cout << "{" << winnerNet.weights[0];
    for (size_t i = 1; i < winnerNet.weights.size(); i++) {
        std::cout << "," << winnerNet.weights[i];
    }
    std::cout << "}" << std::endl;

    return 0;
}