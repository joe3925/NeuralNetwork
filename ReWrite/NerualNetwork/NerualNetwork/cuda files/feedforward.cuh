#include "memoryUtils.cuh"
int feedForwardCUDA(DeviceNetwork deviceNetwork, const std::vector<double>& imageToUse, const std::vector<int>& nodesPerLayers);