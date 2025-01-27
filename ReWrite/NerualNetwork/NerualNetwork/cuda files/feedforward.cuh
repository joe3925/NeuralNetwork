#include "memoryUtils.cuh"
int feedForwardCUDA(DeviceNetwork deviceNetwork, const std::vector<double>& imageToUse, int numLayers, int maxLayerSize);