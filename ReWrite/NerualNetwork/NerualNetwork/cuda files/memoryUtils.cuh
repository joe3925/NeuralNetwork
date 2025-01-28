#include <vector>
#pragma once
#ifndef MEMORYUTILS_CUH
#define MEMORYUTILS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include <iostream>

#include "../helper.h"
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
    double* weights;   // Flattened weights]
    int* layerSizes;   // Number of nodes per layer
    int* targetOutput; //only used to backprop contains the target output of the network
    int totalLayers;   // Total number of layers
};
struct DeviceNode {
    double* weights; 
    double value;
    double bias;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
typedef double (*DeviceActivationFunc)(double);
__device__ DeviceActivationFunc activation = nullptr;
__device__ DeviceActivationFunc activationDerivative = nullptr;

__global__ void setActivationFunctions(DeviceActivationFunc act, DeviceActivationFunc actDeriv);

void setActivation(DeviceActivationFunc act, DeviceActivationFunc actDeriv);

inline int loadNetwork(const std::vector<std::vector<Node>>& network, DeviceNetwork& deviceNetwork) {
    int totalNodes = 0;
    int totalWeights = 0;

    // Calculate total nodes and weights
    for (size_t i = 0; i < network.size(); ++i) {
        totalNodes += network[i].size();
        if (i < network.size() - 1) {
            totalWeights += network[i].size() * network[i + 1].size();
        }
    }

    // Allocate device memory
    CUDA_CHECK_AND_FAULT(cudaMalloc((void**)&deviceNetwork.values, totalNodes * sizeof(double)));
    CUDA_CHECK_AND_FAULT(cudaMalloc((void**)&deviceNetwork.biases, totalNodes * sizeof(double)));
    CUDA_CHECK_AND_FAULT(cudaMalloc((void**)&deviceNetwork.weights, totalWeights * sizeof(double)));
    CUDA_CHECK_AND_FAULT(cudaMalloc((void**)&deviceNetwork.layerSizes, network.size() * sizeof(int)));
    CUDA_CHECK_AND_FAULT(cudaMalloc((void**)&deviceNetwork.targetOutput, network.size() * sizeof(int)));

    // Flatten and copy data to device
    std::vector<double> h_values(totalNodes, 0.0);
    std::vector<double> h_biases(totalNodes, 0.0);
    std::vector<double> h_weights(totalWeights, 0.0);
    std::vector<int> h_layerSizes(network.size());

    int nodeOffset = 0;
    int weightOffset = 0;
    for (size_t i = 0; i < network.size(); ++i) {
        h_layerSizes[i] = network[i].size();

        for (size_t j = 0; j < network[i].size(); ++j) {
            h_values[nodeOffset] = network[i][j].value;
            h_biases[nodeOffset] = network[i][j].bias;

            if (i < network.size() - 1) {
                for (size_t k = 0; k < network[i + 1].size(); ++k) {
                    h_weights[weightOffset] = network[i][j].weights[k];
                    weightOffset++;
                }
            }
            nodeOffset++;
        }
    }

    CUDA_CHECK_AND_FAULT(cudaMemcpy(deviceNetwork.values, h_values.data(), totalNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_FAULT(cudaMemcpy(deviceNetwork.biases, h_biases.data(), totalNodes * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_FAULT(cudaMemcpy(deviceNetwork.weights, h_weights.data(), totalWeights * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_FAULT(cudaMemcpy(deviceNetwork.layerSizes, h_layerSizes.data(), network.size() * sizeof(int), cudaMemcpyHostToDevice));
}

inline int freeNetwork(DeviceNetwork& deviceNetwork) {
    CUDA_CHECK_AND_FAULT(cudaFree(deviceNetwork.values));
    CUDA_CHECK_AND_FAULT(cudaFree(deviceNetwork.biases));
    CUDA_CHECK_AND_FAULT(cudaFree(deviceNetwork.weights));
    CUDA_CHECK_AND_FAULT(cudaFree(deviceNetwork.layerSizes));
    return 0; // Success
}

#endif