#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>
#include "helper.h"

using namespace std::chrono;

using FunctionType = double (*)(double);

class Network
{
private:
    //non accessable
    std::vector<std::vector<double>> tempLayersValuesPreActivation;
    std::vector<std::vector<double>> tempLayersValuesPostActivation;
    std::vector<std::vector<double>> bias;

    std::vector<std::vector<std::vector<double>>> weights;


    //read only
    std::vector<std::vector<double>> layersValuesPreActivation;
    std::vector<std::vector<double>> layersValuesPostActivation;

   
    std::vector<int> nodesPerLayer;

    int layers = 0;
    int timesRan = 0;

    FunctionType activation;
    FunctionType activationDerivative;

    // read and write
    std::vector<int> currentTargetOutput;


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

    void resetNodes()
    {
        for (int i = 0; i < layers; i++)
        {
            std::fill(layersValuesPreActivation[i].begin(), layersValuesPreActivation[i].end(), 0);
            std::fill(layersValuesPostActivation[i].begin(), layersValuesPostActivation[i].end(), 0);
        }
    }
public:


    Network(const std::vector<int>& mNodesPerLayer, FunctionType mActivation, FunctionType mActivationDerivative)
        : nodesPerLayer(mNodesPerLayer),
        layers(mNodesPerLayer.size()),
        activation(mActivation),
        activationDerivative(mActivationDerivative)
    {
    }
    //public variables
    std::vector<int> results;

    //Getters 
    std::vector<std::vector<double>> getLayersValuesPreActivation() const { return layersValuesPreActivation; }
    std::vector<std::vector<double>> getLayersValuesPostActivation() const { return layersValuesPostActivation; }

    std::vector<int> getCurrentTargetOutput() const { return currentTargetOutput; }
    std::vector<int> getNodesPerLayer() const { return nodesPerLayer; }

    int getLayers() const { return layers; }
    int getTimesRan() const { return timesRan; }

    FunctionType getActivation() const { return activation; }
    FunctionType getActivationDerivative() const { return activationDerivative; }

    // Setters
    void setCurrentTargetOutput(const std::vector<int>& newCurrentTargetOutput) {
        currentTargetOutput = newCurrentTargetOutput;
    }


    void HE_initializeWeightsAndBiases()
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < layers - 1; ++i)
        {
            int layerInputs = nodesPerLayer[i];
            int layerOutputs = nodesPerLayer[i + 1];

            std::normal_distribution<> d(0, std::sqrt(2.0 / layerInputs));

            weights.push_back(std::vector<std::vector<double>>(layerOutputs, std::vector<double>(layerInputs)));
            bias.push_back(std::vector<double>(layerOutputs, 0.0));

            for (int j = 0; j < layerOutputs; ++j)
            {
                for (int k = 0; k < layerInputs; ++k)
                {
                    weights[i][j][k] = d(gen);
                }
            }
        }
    }

    void intNodes()
    {
        std::vector<double> temp;
        layersValuesPreActivation.clear();
        for (int i = 0; i < layers; i++)
        {
            layersValuesPreActivation.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i]; j++)
            {
                layersValuesPreActivation[i].push_back(0);
            }
        }
        layersValuesPostActivation.clear();
        for (int i = 0; i < layers; i++)
        {
            layersValuesPostActivation.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i]; j++)
            {
                layersValuesPostActivation[i].push_back(0);
            }
        }
    }

    int xavierIntWeights()
    {
        if (layers == 0 || nodesPerLayer.empty() || layers != nodesPerLayer.size())
        {
            std::cerr << "Layers, nodesPerLayer, or both were initialized wrong. Check parameters of intNetworkObject.";
            return 1;
        }

        weights.clear(); // Clear existing weights

        for (int i = 0; i < layers - 1; ++i)
        { // For each layer
            std::vector<std::vector<double>> layerWeights;
            layerWeights.resize(nodesPerLayer[i + 1]); // Prepare the next layer node weights container

            int prevLayerNodeCount = nodesPerLayer[i];
            int nextLayerNodeCount = nodesPerLayer[i + 1];

            for (int z = 0; z < prevLayerNodeCount; ++z)
            { // For each weight connecting to a node
                for (int j = 0; j < nextLayerNodeCount; ++j)
                { // For each node in the next layer
                    if (z == 0)
                    { // On the first pass, initialize sub-vectors
                        layerWeights[j] = std::vector<double>(prevLayerNodeCount);
                    }
                    layerWeights[j][z] = xavierInitialization(prevLayerNodeCount, nextLayerNodeCount);
                }
            }
            weights.push_back(layerWeights); // Add initialized layer weights to the network
        }
        return 0;
    }

    int intWeights()
    {
        //temp values to resize the vectors since non const cant be passed in for size for some reason
        std::vector<std::vector<double>> temp;
        std::vector<double> temp1;
        //check if values are valid
        if (layers == NULL || nodesPerLayer[0] == NULL || (layers != nodesPerLayer.size()))
        {
            std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check parameters of intNetworkObject";
            return 1;
        }
        for (int i = 0; i < layers - 1; i++)
        {
            weights.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i + 1]; j++)
            {
                weights[i].push_back(temp1);
            }
        }
        return 0;
    }

    int intBiasforImport()
    {
        std::vector<double> temp;
        if (layers == NULL || nodesPerLayer[0] == NULL || (layers != nodesPerLayer.size()))
        {
            std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check constructor or if you imported assure the import wasnt corrupted";
            return 1;
        }
        for (int i = 0; i < layers - 1; i++)
        {
            bias.push_back(temp);
        }
        return 0;
    }

    int intBias()
    {
        std::vector<double> temp;
        if (layers == NULL || nodesPerLayer[0] == NULL || (layers != nodesPerLayer.size()))
        {
            std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check constructor or if you imported assure the import wasnt corrupted";
            return 1;
        }
        for (int i = 0; i < layers - 1; i++)
        {
            bias.push_back(temp);
            for (int j = 0; j < nodesPerLayer[i + 1]; j++)
            {
                bias[i].push_back(0.0);
            }
        }
        return 0;
    }

    //image to use is also used to index the target
    int feedForward(std::vector<double>& imageToUse, bool includeResult)
    {
        resetNodes();

        //put image into input layer
        layersValuesPreActivation[0] = imageToUse;
        layersValuesPostActivation[0] = imageToUse;
        //feed forward
        for (int currentLayer = 0; currentLayer < layers - 1; currentLayer++)
        {
            for (int currentNode = 0; currentNode < nodesPerLayer[currentLayer + 1]; currentNode++)
            {
                for (int currentWeight = 0; currentWeight < weights[currentLayer][currentNode].size(); currentWeight++)
                {
                    layersValuesPreActivation[currentLayer + 1][currentNode] += layersValuesPreActivation[currentLayer][currentWeight] * weights[currentLayer][currentNode][currentWeight];
                }
                layersValuesPreActivation[currentLayer + 1][currentNode] += bias[currentLayer][currentNode];
                layersValuesPostActivation[currentLayer + 1][currentNode] = activation(layersValuesPreActivation[currentLayer + 1][currentNode]);
            }
        }
        //result is the largest value in the output layer
        if (includeResult == true)
        {
            results.push_back(findLargest(layersValuesPostActivation[layersValuesPostActivation.size() - 1]));
        }
        return findLargest(layersValuesPostActivation[layersValuesPostActivation.size() - 1]);
        timesRan++;
    }

    //if a file already exist this WILL overwrite the file
    void exportNetwork(const std::string& networkName)
    {
        std::ofstream exportFile(networkName);
        //save the weights
        for (int i = 0; i < nodesPerLayer.size(); i++)
        {
            exportFile << nodesPerLayer[i];
            exportFile << "\n";
        }
        for (int i = 0; i < weights.size(); i++)
        {
            exportFile << "{";
            for (int j = 0; j < weights[i].size(); j++)
            {
                exportFile << "\n";
                for (int z = 0; z < weights[i][j].size(); z++)
                {
                    exportFile << weights[i][j][z];
                    if (z == weights[i][j].size() - 1)
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
        for (int i = 0; i < bias.size(); i++)
        {
            exportFile << "\n";
            for (int j = 0; j < bias[i].size(); j++)
            {
                exportFile << bias[i][j];
                if (j == bias[i].size() - 1)
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

    void importNetwork(const std::string& path)
    {
        //counters
        int currentLayer = 0;
        int currentNode = 0;
        int currentBiasLayer = 0;

        std::string line;
        std::ifstream file(path);
        getline(file, line);
        while (line != "{")
        {
            nodesPerLayer.push_back(stoi(line));
            getline(file, line);
        }
        intNodes();
        layers = nodesPerLayer.size();
        intBiasforImport();
        intWeights();

        //read accoring to the format of the export
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
                    weights[currentLayer][currentNode].push_back(std::stod(line));
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
                    bias[currentBiasLayer].push_back(stod(line));
                }
            }
        }
    }

    void backPropagate(double learningRate)
    {
        // Calculate output layer error
        std::vector<double> outputLayerError(nodesPerLayer.back());
        for (int i = 0; i < outputLayerError.size(); i++)
        {
            double output = layersValuesPostActivation.back()[i];
            outputLayerError[i] = (2 * (output - currentTargetOutput[i])) * activationDerivative(output);
        }

        // Store the errors for each layer
        std::vector<std::vector<double>> errors(layers);
        errors.back() = outputLayerError;

        // Backpropagate the error
        for (int i = layers - 2; i >= 0; i--)
        {
            errors[i] = std::vector<double>(nodesPerLayer[i], 0.0);
            for (int j = 0; j < nodesPerLayer[i + 1]; j++)
            {
                for (int k = 0; k < nodesPerLayer[i]; k++)
                {
                    errors[i][k] += weights[i][j][k] * errors[i + 1][j];
                }
            }
            for (int j = 0; j < nodesPerLayer[i]; j++)
            {
                errors[i][j] *= activationDerivative(layersValuesPostActivation[i][j]);
            }
        }

        // Update weights and biases
        for (int i = 0; i < layers - 1; i++)
        {
            for (int j = 0; j < nodesPerLayer[i + 1]; j++)
            {
                // Update bias
                bias[i][j] -= learningRate * errors[i + 1][j];
                for (int k = 0; k < nodesPerLayer[i]; k++)
                {
                    // Update weight
                    weights[i][j][k] -= learningRate * layersValuesPostActivation[i][k] * errors[i + 1][j];
                }
            }
        }
    }
    void cleanNetwork()
	{
		layersValuesPreActivation.clear();
		layersValuesPostActivation.clear();
		weights.clear();
		bias.clear();
		nodesPerLayer.clear();
		currentTargetOutput.clear();
		results.clear();
		layers = 0;
		timesRan = 0;
	}

};
