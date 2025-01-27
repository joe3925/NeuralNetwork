#include <vector>
#pragma once
#ifndef MEMORYUTILS_CUH
#define MEMORYUTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
struct Node {
    double value;
    double bias;
    std::vector<double> weights; // Weights of connections to next layer
    Node(int numWeights, double biasInit = 0.0) : value(0), bias(biasInit) {
        for (int i = 0; i < numWeights; i++) {
            weights.push_back(0);
        }
    }
};

struct DeviceNetwork {
    double* values;    // Flattened node values
    double* biases;    // Flattened biases
    double* weights;   // Flattened weights
    int* layerSizes;   // Number of nodes per layer
    int totalLayers;   // Total number of layers
};
struct DeviceNode {
    double* weights; 
    double value;
    double bias;
};

inline int loadNetwork(const std::vector<std::vector<Node>>& network, DeviceNetwork& d_network) {
    // Calculate the total number of nodes and weights
    int totalNodes = 0;
    int totalWeights = 0;
    for (size_t i = 0; i < network.size(); ++i) {
        totalNodes += network[i].size();
        if (i < network.size() - 1) { // Skip weights for the last layer
            totalWeights += network[i].size() * network[i + 1].size();
        }
    }

    // Allocate flattened arrays on the device
    cudaError_t status = cudaMalloc((void**)&d_network.values, totalNodes * sizeof(double));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for values: " << cudaGetErrorString(status) << std::endl;
        return -1;
    }

    status = cudaMalloc((void**)&d_network.biases, totalNodes * sizeof(double));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for biases: " << cudaGetErrorString(status) << std::endl;
        cudaFree(d_network.values);
        return -1;
    }

    status = cudaMalloc((void**)&d_network.weights, totalWeights * sizeof(double));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for weights: " << cudaGetErrorString(status) << std::endl;
        cudaFree(d_network.values);
        cudaFree(d_network.biases);
        return -1;
    }

    status = cudaMalloc((void**)&d_network.layerSizes, network.size() * sizeof(int));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for layerSizes: " << cudaGetErrorString(status) << std::endl;
        cudaFree(d_network.values);
        cudaFree(d_network.biases);
        cudaFree(d_network.weights);
        return -1;
    }

    d_network.totalLayers = static_cast<int>(network.size());

    // Host-side flattened arrays
    std::vector<double> h_values(totalNodes, 0.0);
    std::vector<double> h_biases(totalNodes, 0.0);
    std::vector<double> h_weights(totalWeights, 0.0);
    std::vector<int> h_layerSizes(network.size());

    // Flatten the network into arrays
    int valueIndex = 0;
    int weightIndex = 0;
    for (size_t i = 0; i < network.size(); ++i) {
        h_layerSizes[i] = static_cast<int>(network[i].size());
        for (const Node& node : network[i]) {
            h_values[valueIndex] = node.value;
            h_biases[valueIndex] = node.bias;
            if (i < network.size() - 1) { // Add weights for intermediate layers
                for (double weight : node.weights) {
                    h_weights[weightIndex++] = weight;
                }
            }
            ++valueIndex;
        }
    }

    // Copy flattened arrays to device memory
    cudaMemcpy(d_network.values, h_values.data(), totalNodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network.biases, h_biases.data(), totalNodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network.weights, h_weights.data(), totalWeights * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network.layerSizes, h_layerSizes.data(), network.size() * sizeof(int), cudaMemcpyHostToDevice);

    return 0; // Success
}

inline int freeNetwork(DeviceNetwork& d_network) {
    cudaFree(d_network.values);
    cudaFree(d_network.biases);
    cudaFree(d_network.weights);
    cudaFree(d_network.layerSizes);

    d_network.values = nullptr;
    d_network.biases = nullptr;
    d_network.weights = nullptr;
    d_network.layerSizes = nullptr;

    return 0; // Success
}

#endif