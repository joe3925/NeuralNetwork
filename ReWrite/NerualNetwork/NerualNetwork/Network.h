#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>


using namespace std::chrono;
double xavierInitialization(int fan_in, int fan_out);
double sigmoid(double m1);

template <typename T>
int findLargest(std::vector<T>& vec);
template <typename T>
int findSmallestPosition(const std::vector<T>& vec);

using FunctionType = double (*)(double);


class Network
{

private:

public:
    std::vector<int> nodesPerLayer;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<double>> layersValuesPreActivation;
    std::vector<std::vector<double>> layersValuesPostActivation;
    std::vector<std::vector<double>> tempLayersValuesPreActivation;
    std::vector<std::vector<double>> tempLayersValuesPostActivation;
    int layers = 0;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<int> results;
    std::vector<int> currentTargetOutput;
    FunctionType activation;
    FunctionType activationDerivative;
    

    int timesRan = 0;
    Network(const std::vector<int>& nodesPerLayer1,
        FunctionType activation1, FunctionType activationDerivative1) // Add the function pointer parameter
        : nodesPerLayer(nodesPerLayer1),
        layers(nodesPerLayer1.size()),
        activation(activation1),
        activationDerivative(activationDerivative1)
    {    }
    void HE_initializeWeightsAndBiases() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < layers - 1; ++i) {
            int layerInputs = nodesPerLayer[i];
            int layerOutputs = nodesPerLayer[i + 1];

            std::normal_distribution<> d(0, std::sqrt(2.0 / layerInputs));

            weights.push_back(std::vector<std::vector<double>>(layerOutputs, std::vector<double>(layerInputs)));
            bias.push_back(std::vector<double>(layerOutputs, 0.0));

            for (int j = 0; j < layerOutputs; ++j) {
                for (int k = 0; k < layerInputs; ++k) {
                    weights[i][j][k] = d(gen);
                }
            }
        }
    }
 
};
void createTempWeights(Network& network)
{
    std::vector<double> temp;
    network.tempLayersValuesPreActivation.clear();
    for (int i = 0; i < network.layers; i++)
    {
        network.tempLayersValuesPreActivation.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i]; j++)
        {
            network.tempLayersValuesPreActivation[i].push_back(0);
        }
    }
    network.tempLayersValuesPostActivation.clear();
    for (int i = 0; i < network.layers; i++)
    {
        network.tempLayersValuesPostActivation.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i]; j++)
        {
            network.tempLayersValuesPostActivation[i].push_back(0);
        }
    }
}

void intNodes(Network& network)
{
    std::vector<double> temp;
    network.layersValuesPreActivation.clear();
    for (int i = 0; i < network.layers; i++)
    {
        network.layersValuesPreActivation.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i]; j++)
        {
            network.layersValuesPreActivation[i].push_back(0);
        }
    }
    network.layersValuesPostActivation.clear();
    for (int i = 0; i < network.layers; i++)
    {
        network.layersValuesPostActivation.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i]; j++)
        {
            network.layersValuesPostActivation[i].push_back(0);
        }
    }
    createTempWeights(network);
}
void resetNodes(Network& network)
{
    network.layersValuesPostActivation = network.tempLayersValuesPostActivation;
    network.layersValuesPreActivation = network.tempLayersValuesPreActivation;

}


int xavierIntWeights(Network& network)
{
    if (network.layers == 0 || network.nodesPerLayer.empty() || network.layers != network.nodesPerLayer.size()) {
        std::cerr << "Layers, nodesPerLayer, or both were initialized wrong. Check parameters of intNetworkObject.";
        return 1;
    }

    network.weights.clear(); // Clear existing weights

    for (int i = 0; i < network.layers - 1; ++i) { // For each layer
        std::vector<std::vector<double>> layerWeights;
        layerWeights.resize(network.nodesPerLayer[i + 1]); // Prepare the next layer node weights container

        int prevLayerNodeCount = network.nodesPerLayer[i];
        int nextLayerNodeCount = network.nodesPerLayer[i + 1];

        for (int z = 0; z < prevLayerNodeCount; ++z) { // For each weight connecting to a node
            for (int j = 0; j < nextLayerNodeCount; ++j) { // For each node in the next layer
                if (z == 0) { // On the first pass, initialize sub-vectors
                    layerWeights[j] = std::vector<double>(prevLayerNodeCount);
                }
                layerWeights[j][z] = xavierInitialization(prevLayerNodeCount, nextLayerNodeCount);
            }
        }
        network.weights.push_back(layerWeights); // Add initialized layer weights to the network
    }
    return 0;
}

int intWeights(Network& network)
{
    //temp values to resize the vectors since non const cant be passed in for size for some reason
    std::vector<std::vector<double>> temp;
    std::vector<double> temp1;
    //check if values are valid
    if (network.layers == NULL || network.nodesPerLayer[0] == NULL || (network.layers != network.nodesPerLayer.size()))
    {
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check parameters of intNetworkObject";
        return 1;
    }
    for (int i = 0; i < network.layers - 1; i++)
    {
        network.weights.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++)
        {
            network.weights[i].push_back(temp1);
        }
    }
    return 0;
}

int intBiasforImport(Network& network)
{
    std::vector<double> temp;
    if (network.layers == NULL || network.nodesPerLayer[0] == NULL || (network.layers != network.nodesPerLayer.size()))
    {
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check constructor or if you imported assure the import wasnt corrupted";
        return 1;
    }
    for (int i = 0; i < network.layers - 1; i++)
    {
        network.bias.push_back(temp);
    }
    return 0;
}


int intBias(Network& network)
{
    std::vector<double> temp;
    if (network.layers == NULL || network.nodesPerLayer[0] == NULL || (network.layers != network.nodesPerLayer.size()))
    {
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check constructor or if you imported assure the import wasnt corrupted";
        return 1;
    }
    for (int i = 0; i < network.layers - 1; i++)
    {
        network.bias.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++)
        {
            network.bias[i].push_back(0.0);
        }
    }
    return 0;
}


//image to use is also used to index the target
int feedForward(Network& network, std::vector<double>& imageToUse, bool includeResult)
{

    resetNodes(network);

    //put image into input layer
    network.layersValuesPreActivation[0] = imageToUse;
    network.layersValuesPostActivation[0] = imageToUse;
    //feed forward
    for (int currentLayer = 0; currentLayer < network.layers - 1; currentLayer++)
    {
        for (int currentNode = 0; currentNode < network.nodesPerLayer[currentLayer + 1]; currentNode++)
        {
            for (int currentWeight = 0; currentWeight < network.weights[currentLayer][currentNode].size(); currentWeight++)
            {
                network.layersValuesPreActivation[currentLayer + 1][currentNode] += network.layersValuesPreActivation[currentLayer][currentWeight] * network.weights
                    [currentLayer][currentNode][currentWeight];
            }
            network.layersValuesPreActivation[currentLayer + 1][currentNode] += network.bias[currentLayer][currentNode];
            network.layersValuesPostActivation[currentLayer + 1][currentNode] = network.activation(network.layersValuesPreActivation[currentLayer + 1][currentNode]);
        }
    }
    //increment times ran and give an answer 
    //result is the largest value in the output layer
    if (includeResult == true) {
        network.results.push_back(
            findLargest(network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1]));
    }
    return findLargest(network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1]);
    network.timesRan++;
}

//if a file already exist this WILL overwrite the file
void exportNetwork(const std::string& networkName, Network& network)
{
    std::ofstream exportFile(networkName);
    //save the weights 
    for (int i = 0; i < network.nodesPerLayer.size(); i++)
    {
        exportFile << network.nodesPerLayer[i];
        exportFile << "\n";
    }
    for (int i = 0; i < network.weights.size(); i++)
    {
        exportFile << "{";
        for (int j = 0; j < network.weights[i].size(); j++)
        {
            exportFile << "\n";
            for (int z = 0; z < network.weights[i][j].size(); z++)
            {
                exportFile << network.weights[i][j][z];
                if (z == network.weights[i][j].size() - 1)
                {
                    exportFile << "\n";
                    exportFile << ";";
                    break;
                }
                exportFile << "\n";
            }
        }
        exportFile << "}";
        exportFile << "\n";
    }
    exportFile << "(";
    //save bias
    for (int i = 0; i < network.bias.size(); i++)
    {
        exportFile << "\n";
        for (int j = 0; j < network.bias[i].size(); j++)
        {
            exportFile << network.bias[i][j];
            if (j == network.bias[i].size() - 1)
            {
                exportFile << "\n";
                exportFile << ";";
                break;
            }
            exportFile << "\n";
        }
    }
    exportFile << ")";
}

void importNetwork(const std::string& path, Network& network)
{
    std::string line;
    std::ifstream file(path);
    std::vector<std::vector<double>> temp;
    std::vector<double> temp1;
    int currentLayer = 0;
    int currentNode = 0;
    int currentBiasLayer = 0;
    getline(file, line);
    while (line != "{")
    {
        network.nodesPerLayer.push_back(stoi(line));
        getline(file, line);
    }
    intNodes(network);
    network.layers = network.nodesPerLayer.size();
    intBiasforImport(network);
    intWeights(network);

    //rea
    if (line == "{")
    {
        while (line != "(")
        {
            getline(file, line);
            if (line == "}")
            {
                currentLayer++;
            }
            else if (line == ";")
            {
                currentNode++;
            }
            else if (line == ";}")
            {
                currentLayer++;
                currentNode = 0;
                getline(file, line);
            }
            else
            {
                network.weights[currentLayer][currentNode].push_back(std::stod(line));
            }
        }
        while (true)
        {
            getline(file, line);
            if (line == ";")
            {
                currentBiasLayer++;
            }
            else if (line == ";)")
            {
                break;
            }
            else
            {
                network.bias[currentBiasLayer].push_back(stod(line));
            }
        }
    }
}

double xavierInitialization(int fan_in, int fan_out) {
    // Use Xavier initialization (Glorot initialization)
    double variance = 1.0 / (fan_in + fan_out);
    double stddev = sqrt(variance);

    // Seed with a real random value, if available
    std::random_device rd;

    // Use a high-resolution clock to ensure different seeds when called in quick succession
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Combine seed from high-resolution clock and random_device
    std::seed_seq seq{ rd(), static_cast<unsigned int>(seed) };
    std::mt19937 generator(seq);

    std::normal_distribution<double> distribution(0.0, stddev);

    return distribution(generator);
}

double sigmoid(double m1)
{
    return 1 / (1 + exp(-m1));
}

double sigmoidDerivative(double m1)
{

    double solve = exp(m1) / ((exp(m1) + 1) * (exp(m1) + 1));
    return solve;

}

double costDerivative (double expected, double result){
    return 2*(result - expected);
}

void backPropagate(Network& network, double learningRate) {
    // Calculate output layer error
    std::vector<double> outputLayerError(network.nodesPerLayer.back());
    for (int i = 0; i < outputLayerError.size(); i++) {
        double output = network.layersValuesPostActivation.back()[i];
        outputLayerError[i] = (output - network.currentTargetOutput[i]) * network.activationDerivative(output);
    }

    // Store the errors for each layer
    std::vector<std::vector<double>> errors(network.layers);
    errors.back() = outputLayerError;

    // Backpropagate the error
    for (int i = network.layers - 2; i >= 0; i--) {
        errors[i] = std::vector<double>(network.nodesPerLayer[i], 0.0);
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++) {
            for (int k = 0; k < network.nodesPerLayer[i]; k++) {
                errors[i][k] += network.weights[i][j][k] * errors[i + 1][j];
            }
        }
        for (int j = 0; j < network.nodesPerLayer[i]; j++) {
            errors[i][j] *= network.activationDerivative(network.layersValuesPostActivation[i][j]);
        }
    }

    // Update weights and biases
    for (int i = 0; i < network.layers - 1; i++) {
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++) {
            // Update bias
            network.bias[i][j] -= learningRate * errors[i + 1][j];
            for (int k = 0; k < network.nodesPerLayer[i]; k++) {
                // Update weight
                network.weights[i][j][k] -= learningRate * network.layersValuesPostActivation[i][k] * errors[i + 1][j];
            }
        }
    }
}


template <typename T>
int findLargest(std::vector<T>& vec)
{
    if (vec.empty())
    {
        // Handle the case when the vector is empty, or you can return an error code.
        return -1; // Return -1 to indicate an error.
    }

    double maxVal = vec[0];
    int position = 0;

    for (int i = 1; i < vec.size(); i++)
    {
        if (vec[i] > maxVal)
        {
            maxVal = vec[i];
            position = i;
        }
    }

    return position;
}
template <typename T>
int findSmallestPosition(const std::vector<T>& vec) {
    if (vec.empty()) {
        // Handle the case where the vector is empty
        std::cout << "Vector is empty." << std::endl;
        return -1; // Return an invalid position
    }

    int smallestPosition = 0;
    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[smallestPosition]) {
            smallestPosition = i;
        }
    }

    return smallestPosition;
}