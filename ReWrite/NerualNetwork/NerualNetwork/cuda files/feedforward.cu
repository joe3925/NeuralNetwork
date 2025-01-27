#include <cuda_runtime.h>
#include <cmath>
#include "feedforward.cuh"
//TODO: change this to be modular 
__device__ double activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ void computeLayerFlat(DeviceNetwork deviceNetwork, int currentLayer, int nextLayerSize, int currentLayerSize, int maxLayerSize) {
    int nextNodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nextNodeIdx >= nextLayerSize) return;

    double weightedSum = 0.0;

    // Calculate weighted sum for the next node
    for (int currentNodeIdx = 0; currentNodeIdx < currentLayerSize; ++currentNodeIdx) {
        int weightIdx = currentLayer * maxLayerSize * maxLayerSize + currentNodeIdx * maxLayerSize + nextNodeIdx;
        weightedSum += deviceNetwork.values[currentLayer * maxLayerSize + currentNodeIdx] *
            deviceNetwork.weights[weightIdx];
    }

    // Calculate the value for the next node
    int nextNodeIndex = (currentLayer + 1) * maxLayerSize + nextNodeIdx;
    deviceNetwork.values[nextNodeIndex] = activation(weightedSum + deviceNetwork.biases[nextNodeIndex]);
}


int feedForwardCUDA(DeviceNetwork deviceNetwork, const std::vector<double>& imageToUse, int numLayers, int maxLayerSize) {
    // Copy input data to device values array (input layer values)
    cudaMemcpy(deviceNetwork.values, imageToUse.data(), imageToUse.size() * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;

    // Perform feedforward operation layer by layer
    for (int currentLayer = 0; currentLayer < numLayers - 1; ++currentLayer) {
        int currentLayerSize = deviceNetwork.layerSizes[currentLayer];
        int nextLayerSize = deviceNetwork.layerSizes[currentLayer + 1];

        int blocksPerGrid = (nextLayerSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel for the current layer
        computeLayerFlat << <blocksPerGrid, threadsPerBlock >> > (
            deviceNetwork, currentLayer, nextLayerSize, currentLayerSize, maxLayerSize);
        cudaDeviceSynchronize(); // Ensure kernel execution is completed before moving to the next layer
    }

    // Copy output values from the last layer back to the host
    int outputLayerSize = deviceNetwork.layerSizes[numLayers - 1];
    std::vector<double> outputValues(outputLayerSize);
    cudaMemcpy(outputValues.data(), deviceNetwork.values + (numLayers - 1) * maxLayerSize,
        outputLayerSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Determine the index of the maximum value in the output layer (argmax operation)
    int resultIndex = 0;
    double maxValue = outputValues[0];
    for (int i = 1; i < outputLayerSize; ++i) {
        if (outputValues[i] > maxValue) {
            maxValue = outputValues[i];
            resultIndex = i;
        }
    }

    return resultIndex;
}
