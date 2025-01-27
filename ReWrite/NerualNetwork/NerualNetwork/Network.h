#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>
#include "helper.h"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include "cuda files/feedforward.cuh"
#
using namespace std::chrono;

using activationFunc = double (*)(double);




class Network
{
public:
    std::vector<std::vector<Node>> network;

private:

    std::vector<std::vector<double>> tempLayersValuesPreActivation;
    std::vector<std::vector<double>> tempLayersValuesPostActivation;
    std::vector<std::vector<double>> bias;


    std::vector<int> nodesPerLayer;
    int layers = 0;
    activationFunc activation;
    activationFunc activationDerivative;


    std::vector<int> currentTargetOutput;


    int timesRan = 0;

    bool mUse_gpu;
    DeviceNetwork device_network;

    double xavierInitialization(int fan_in, int fan_out)
    {
        // Use Xavier initialization
        double variance = 1.0 / (fan_in + fan_out);
        double stddev = sqrt(variance);

        std::random_device rd;

        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        // Combine seed from high-resolution clock and random_device
        std::seed_seq seq{ rd(), static_cast<unsigned int>(seed) };
        std::mt19937 generator(seq);

        std::normal_distribution<double> distribution(0.0, stddev);

        return distribution(generator);
    }

    static double costDerivative(double expected, double result)
    {
        return 2 * (result - expected);
    }
public:

    Network(const std::vector<int>& nodesPerLayer, bool use_gpu, activationFunc act, activationFunc actDeriv)
        : activation(act), activationDerivative(actDeriv), nodesPerLayer(nodesPerLayer) {
        // Initialize layers
        std::vector<Node> layer;
        for (size_t i = 0; i < nodesPerLayer.size(); ++i) {
            int numWeights = (i < nodesPerLayer.size() - 1) ? nodesPerLayer[i + 1] : 0;
            network.emplace_back(nodesPerLayer[i], Node(numWeights));
        }
        if (use_gpu) {
            mUse_gpu = true;
            loadNetwork(network, device_network);
        }
        else {
            mUse_gpu = false;
        }
    }
    //public variables
    std::vector<int> results;

    std::vector<int> getCurrentTargetOutput() const { return currentTargetOutput; }
    std::vector<int> getNodesPerLayer() const { return nodesPerLayer; }

    int getLayers() const { return layers; }
    int getTimesRan() const { return timesRan; }

    activationFunc getActivation() const { return activation; }
    activationFunc getActivationDerivative() const { return activationDerivative; }

    // Setters
    void setCurrentTargetOutput(const std::vector<int>& newCurrentTargetOutput) {
        currentTargetOutput = newCurrentTargetOutput;
    }

    void HE_initializeWeightsAndBiases() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < network.size() - 1; ++i) {
            int layerInputs = network[i].size();      // Number of nodes in the current layer
            int layerOutputs = network[i + 1].size(); // Number of nodes in the next layer

            // He initialization: normal distribution with stddev = sqrt(2 / inputs)
            std::normal_distribution<> d(0, std::sqrt(2.0 / layerInputs));

            for (size_t j = 0; j < network[i].size(); ++j) {
                // Resize the weights of each node in the current layer to match the number of nodes in the next layer
                network[i][j].weights.resize(layerOutputs);

                for (size_t k = 0; k < layerOutputs; ++k) {
                    // Initialize weights for connections to each node in the next layer
                    network[i][j].weights[k] = d(gen);
                }
            }

            // Initialize biases for the nodes in the next layer
            for (Node& node : network[i + 1]) {
                node.bias = d(gen); // Bias is initialized separately for each node
            }
        }
    }
    int xavierIntWeightsAndBias() {
        if (network.empty()) {
            std::cerr << "Network structure is empty. Ensure nodes are properly initialized.";
            return -1;
        }

        for (size_t i = 0; i < network.size() - 1; ++i) { // Iterate through layers (except the last one)
            int prevLayerNodeCount = network[i].size();      // Number of nodes in the current layer
            int nextLayerNodeCount = network[i + 1].size();  // Number of nodes in the next layer

            // Xavier initialization: variance = 1 / (fan_in + fan_out)
            double variance = 1.0 / (prevLayerNodeCount + nextLayerNodeCount);
            double stddev = std::sqrt(variance);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dist(0.0, stddev);

            // Initialize weights and biases
            for (size_t j = 0; j < network[i].size(); ++j) { // For each node in the current layer
                network[i][j].weights.resize(nextLayerNodeCount); // Resize weights to match next layer size
                for (size_t k = 0; k < nextLayerNodeCount; ++k) {
                    network[i][j].weights[k] = dist(gen); // Initialize weight to node k in the next layer
                }
            }

            for (Node& node : network[i + 1]) { // For each node in the next layer
                node.bias = dist(gen); // Initialize bias for the node
            }
        }
        return 0;
    }


    int feedForward(const std::vector<double>& imageToUse, bool includeResult) {
        if (network.empty() || network[0].size() != imageToUse.size()) {
            std::cerr << "Input size does not match the number of nodes in the input layer.";
            return -1; // Error code
        }
        if (mUse_gpu) {
            return feedForwardCUDA(device_network, imageToUse, nodesPerLayer.size(), findLargest(nodesPerLayer));
        }

        for (auto& layer : network) {
            for (auto& node : layer) {
                node.value = 0.0;
            }
        }

        for (size_t i = 0; i < imageToUse.size(); ++i) {
            network[0][i].value = imageToUse[i];
        }

        for (size_t currentLayer = 0; currentLayer < network.size() - 1; ++currentLayer) {
            for (size_t currentNode = 0; currentNode < network[currentLayer].size(); ++currentNode) {
                const Node& currentNodeObj = network[currentLayer][currentNode];
                for (size_t nextNode = 0; nextNode < network[currentLayer + 1].size(); ++nextNode) {
                    // Add weighted value to the next node
                    network[currentLayer + 1][nextNode].value += currentNodeObj.value * currentNodeObj.weights[nextNode];
                }
            }

            // Apply activation function and add bias to the next layer
            for (Node& nextNode : network[currentLayer + 1]) {
                nextNode.value = activation(nextNode.value + nextNode.bias);
            }
        }

        const std::vector<Node>& outputLayer = network.back();
        int resultIndex = 0;
        double maxValue = outputLayer[0].value;

        for (size_t i = 1; i < outputLayer.size(); ++i) {
            if (outputLayer[i].value > maxValue) {
                maxValue = outputLayer[i].value;
                resultIndex = static_cast<int>(i);
            }
        }

        if (includeResult) {
            results.push_back(resultIndex);
        }

        timesRan++;
        return resultIndex;
    }


    void exportNetwork(const std::string& networkName) {
        std::ofstream exportFile(networkName);
        if (!exportFile.is_open()) {
            std::cerr << "Failed to open file for export: " << networkName << std::endl;
            return;
        }

        // Save the number of nodes per layer
        for (const auto& layer : network) {
            exportFile << layer.size() << "\n";
        }

        // Save weights and biases
        exportFile << "{\n";
        for (size_t i = 0; i < network.size() - 1; ++i) {
            for (size_t j = 0; j < network[i].size(); ++j) {
                exportFile << "["; // Begin weights for the current node
                for (size_t k = 0; k < network[i][j].weights.size(); ++k) {
                    exportFile << network[i][j].weights[k];
                    if (k < network[i][j].weights.size() - 1) {
                        exportFile << ", ";
                    }
                }
                exportFile << "]\n"; // End weights for the current node
            }
        }
        exportFile << "}\n";

        exportFile << "(\n";
        for (size_t i = 1; i < network.size(); ++i) { // Start from layer 1 (biases are not in input layer)
            for (const Node& node : network[i]) {
                exportFile << node.bias << "\n";
            }
        }
        exportFile << ")\n";

        exportFile.close();
    }


    void importNetwork(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for import: " << path << std::endl;
            return;
        }

        network.clear(); // Clear existing network structure

        std::string line;
        std::vector<int> nodesPerLayer;

        // Read nodes per layer
        while (std::getline(file, line) && line != "{") {
            nodesPerLayer.push_back(std::stoi(line));
        }

        // Initialize network structure
        for (int numNodes : nodesPerLayer) {
            network.emplace_back();
            for (int i = 0; i < numNodes; ++i) {
                network.back().emplace_back(numNodes > 1 ? nodesPerLayer.back() : 0);
            }
        }

        // Read weights
        size_t currentLayer = 0;
        size_t currentNode = 0;

        if (line == "{") {
            while (std::getline(file, line) && line != "}") {
                if (line[0] == '[') {
                    line = line.substr(1, line.size() - 2); // Remove brackets
                    std::stringstream ss(line);
                    std::string weight;
                    while (std::getline(ss, weight, ',')) {
                        network[currentLayer][currentNode].weights.push_back(std::stod(weight));
                    }
                    currentNode++;
                    if (currentNode >= network[currentLayer].size()) {
                        currentNode = 0;
                        currentLayer++;
                    }
                }
            }
        }

        // Read biases
        currentLayer = 1; // Skip input layer (no biases)
        if (std::getline(file, line) && line == "(") {
            while (std::getline(file, line) && line != ")") {
                for (Node& node : network[currentLayer]) {
                    node.bias = std::stod(line);
                    if (!std::getline(file, line) || line == ")") break;
                }
                currentLayer++;
            }
        }

        file.close();
    }

    void backPropagate(double learningRate) {
        if (currentTargetOutput.size() != network.back().size()) {
            std::cerr << "Target output size does not match the number of output nodes." << std::endl;
            return;
        }

        // Calculate output layer error
        std::vector<double> outputLayerError(network.back().size());
        for (size_t i = 0; i < network.back().size(); ++i) {
            double output = network.back()[i].value;
            outputLayerError[i] = (2 * (output - currentTargetOutput[i])) * activationDerivative(output);
        }

        // Store the errors for each layer
        std::vector<std::vector<double>> errors(network.size());
        errors.back() = outputLayerError;

        // Backpropagate the error
        for (int layerIdx = network.size() - 2; layerIdx >= 0; --layerIdx) {
            errors[layerIdx].resize(network[layerIdx].size(), 0.0);

            for (size_t nodeIdx = 0; nodeIdx < network[layerIdx].size(); ++nodeIdx) {
                for (size_t nextNodeIdx = 0; nextNodeIdx < network[layerIdx + 1].size(); ++nextNodeIdx) {
                    errors[layerIdx][nodeIdx] +=
                        network[layerIdx][nodeIdx].weights[nextNodeIdx] * errors[layerIdx + 1][nextNodeIdx];
                }
                errors[layerIdx][nodeIdx] *= activationDerivative(network[layerIdx][nodeIdx].value);
            }
        }

        // Update weights and biases
        for (size_t layerIdx = 0; layerIdx < network.size() - 1; ++layerIdx) {
            for (size_t nodeIdx = 0; nodeIdx < network[layerIdx].size(); ++nodeIdx) {
                for (size_t nextNodeIdx = 0; nextNodeIdx < network[layerIdx + 1].size(); ++nextNodeIdx) {
                    // Update weight
                    network[layerIdx][nodeIdx].weights[nextNodeIdx] -=
                        learningRate * network[layerIdx][nodeIdx].value * errors[layerIdx + 1][nextNodeIdx];
                }
            }

            for (size_t nextNodeIdx = 0; nextNodeIdx < network[layerIdx + 1].size(); ++nextNodeIdx) {
                // Update bias
                network[layerIdx + 1][nextNodeIdx].bias -= learningRate * errors[layerIdx + 1][nextNodeIdx];
            }
        }
    }

    void cleanNetwork() {
        int sum = 0;
        for (int i = 0; i < nodesPerLayer.size(); i++) {
            sum += nodesPerLayer[i];
        }
        freeNetwork(device_network);
        timesRan = 0;
        for (auto& layer : network) {
            for (auto& node : layer) {
                node.value = 0.0;
                node.bias = 0.0;
                std::fill(node.weights.begin(), node.weights.end(), 0.0);
            }
        }
        currentTargetOutput.clear();
        results.clear();

    }

};
