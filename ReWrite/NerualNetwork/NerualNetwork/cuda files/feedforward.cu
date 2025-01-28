#include <cuda_runtime.h>
#include <cmath>
#include "feedforward.cuh"
#include "../helper.h"

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
int feedForwardCUDA(DeviceNetwork deviceNetwork, const std::vector<double>& imageToUse, const std::vector<int>& nodesPerLayers) {
    // Copy input data to device values array (input layer values)
    CUDA_CHECK_AND_FAULT(cudaMemcpy(deviceNetwork.values, imageToUse.data(), imageToUse.size() * sizeof(double), cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int numLayers = nodesPerLayers.size();
    int maxLayerSize = findLargest(nodesPerLayers);
    // Perform feedforward operation layer by layer
    for (int currentLayer = 0; currentLayer < numLayers - 1; ++currentLayer) {
        int currentLayerSize = nodesPerLayers[currentLayer];
        int nextLayerSize = nodesPerLayers[currentLayer + 1];

        int blocksPerGrid = (nextLayerSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel for the current layer
        computeLayerFlat << <blocksPerGrid, threadsPerBlock >> > (
            deviceNetwork, currentLayer, nextLayerSize, currentLayerSize, maxLayerSize);
        CUDA_CHECK_AND_FAULT(cudaDeviceSynchronize()); // Ensure kernel execution is completed before moving to the next layer
    }

    // Copy output values from the last layer back to the host
    int outputLayerSize = nodesPerLayers[numLayers - 1];
    std::vector<double> outputValues(outputLayerSize);
    CUDA_CHECK_AND_FAULT(cudaMemcpy(outputValues.data(), deviceNetwork.values + (numLayers - 1) * maxLayerSize,
        outputLayerSize * sizeof(double), cudaMemcpyDeviceToHost));



    return findLargest(outputValues);
}
