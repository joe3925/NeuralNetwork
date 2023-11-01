#pragma once
#include <vector>
#include <random>
#include <chrono>
#include <thread>

using namespace std::chrono;
double xavierInitialization(int fan_in, int fan_out);
class Network
{
public:
    
    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    std::vector<int> nodesPerLayer;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<double>> layersValues;
    size_t layers;
    std::vector<std::vector<std::vector<double>>> weights;

    //recomended to just call intNetwork instead of setting values on your own
    int intNetworkObject(std::vector<std::vector<double>> images1, std::vector<int> labels1, std::vector<int> nodesPerLayer1, Network& network)
    {
        std::vector<double> temp;
        network.nodesPerLayer = nodesPerLayer1;
        network.layers = nodesPerLayer1.size();
        network.images = images1;
        network.labels = labels1;
        for (int i = 0; i < network.layers; i++)
        {
            layersValues.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i]; j++)
            {
                layersValues[i].push_back(0);
            }
        }
        return 0;
    }
    
    int xavierIntWeights(Network& network)
    {
        //temp values to resize the vectors since non const cant be passed in for size for some reason
        std::vector<std::vector<double>> temp;
        std::vector<double> temp1;
        //check if values are valid
        if (network.layers == NULL || network.nodesPerLayer[0] == NULL || (network.layers != network.nodesPerLayer.size()))
        {
            return 1;
        }
        for (int i = 0; i < network.layers - 1; i++)
        {
            network.weights.push_back(temp);
            for(int j = 0; j < network.nodesPerLayer[i + 1]; j++)
            {
                network.weights[i].push_back(temp1);
                for(int z = 0; z < network.nodesPerLayer[i]; z++)
                {
                    network.weights[i][j].push_back(xavierInitialization(nodesPerLayer[i], nodesPerLayer[i + 1]));
                }
            }
        }
        return 0;
    }
    int intBias(Network& network)
    {
       std::vector<double> temp;
        if (network.layers == NULL || network.nodesPerLayer[0] == NULL || (network.layers != network.nodesPerLayer.size()))
        {
            return 1;
        }
        for (int i = 0; i < network.layers - 1; i++ )
        {
            network.bias.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i + 1]; j++)
            {
                network.bias[i].push_back(0.0);
            }
        }
        return 0;
    }
};
double xavierInitialization(int fan_in, int fan_out) {
    // Use Xavier initialization (Glorot initialization)
    auto start = std::chrono::high_resolution_clock::now();
    double variance = 1.0 / (fan_in + fan_out);
    double stddev = sqrt(variance);

    // Generate random values with mean 0 and the calculated standard deviation
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0, stddev);
    high_resolution_clock::time_point startTime = high_resolution_clock::now();
    while ((std::chrono::high_resolution_clock::now() - start).count() < 0) {
        int i = 1;
        i = 0;
    }
    return distribution(generator);
}
