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
double findLargest(std::vector<double> vec);

class Network
{
public:
    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    std::vector<int> nodesPerLayer;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<double>> layersValuesPreActivation;
    std::vector<std::vector<double>> layersValuesPostActivation;
    size_t layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<int> results;
    std::vector<int> currentTargetOutput;
    int timesRan;

    //recommended to just call intNetwork instead of setting values on your own. 

    
    
};
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
}

int intNetworkObject(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::vector<int> nodesPerLayer1, Network& network)
{
    std::vector<double> temp;
    network.nodesPerLayer = nodesPerLayer1;
    network.layers = nodesPerLayer1.size();
    network.images = images1;
    network.labels = labels1;
    intNodes(network);
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
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check parameters of intNetworkObject";
        return 1;
    }
    for (int i = 0; i < network.layers - 1; i++)
    {
        network.weights.push_back(temp);
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++)
        {
            network.weights[i].push_back(temp1);
            for (int z = 0; z < network.nodesPerLayer[i]; z++)
            {
                network.weights[i][j].push_back(xavierInitialization(network.nodesPerLayer[i], network.nodesPerLayer[i + 1]));
            }
        }
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
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check parameters of intNetworkObject";
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
        std::cerr << "Layers, nodesPerLayer or both were initialized wrong. Check parameters of intNetworkObject";
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
void feedForward(Network& network, int imageToUse)
{
    intNodes(network);
    if (network.nodesPerLayer[0] == network.images.size())
    {
        return;
    }
    //put image into input layer
    network.layersValuesPreActivation[0] = network.images[imageToUse];
    network.layersValuesPostActivation[0] = network.images[imageToUse];
    //feed forward
    for (int i = 0; i < network.layers - 1; i++)
    {
        for (int j = 0; j < network.nodesPerLayer[i + 1]; j++)
        {
            for (int z = 0; z < network.weights[i][j].size(); z++)
            {
                network.layersValuesPreActivation[i + 1][j] += network.layersValuesPreActivation[i][z] * network.weights[i][j][z];
            }
            network.layersValuesPreActivation[i + 1][j] += network.bias[i][j];
            network.layersValuesPostActivation[i + 1][j] = sigmoid(network.layersValuesPreActivation[i + 1][j]);

        }
    }
    //increment times ran and give an answer 
    //result is the largest value in the output layer
    network.results.push_back(findLargest(network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1]));
    network.timesRan++;

}
//if a file already exist this WILL overwrite the file
void exportNetwork(std::string networkName, Network& network) {
    std::ofstream exportFile(networkName);
    //save the weights 
    for (int i = 0; i < network.nodesPerLayer.size(); i++) {
        exportFile << network.nodesPerLayer[i];
        exportFile << "\n";
    }
    for (int i = 0; i < network.weights.size(); i++) {
        exportFile << "{";
        for (int j = 0; j < network.weights[i].size(); j++) {
            exportFile << "\n";
            for (int z = 0; z < network.weights[i][j].size(); z++) {
                exportFile << network.weights[i][j][z];
                if (z == network.weights[i][j].size() - 1) {
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
    for (int i = 0; i < network.bias.size(); i++) {
        exportFile << "\n";
        for (int j = 0; j < network.bias[i].size(); j++) {
            exportFile << network.bias[i][j];
            if (j == network.bias[i].size() - 1) {
                exportFile << "\n";
                exportFile << ";";
                break;
            }
            exportFile << "\n";
        }
        
    }
    exportFile << ")";

}
void importNetwork(std::string path, Network& network) {
    std::string line;
    std::ifstream file(path); 
    std::vector<std::vector<double>> temp;
    std::vector<double> temp1;
    int currentLayer = 0;
    int currentNode = 0;
    int currentBiasLayer = 0;
    getline(file, line);
    while (line != "{") {
        network.nodesPerLayer.push_back(stoi(line));
        getline(file, line);

    }
    intNodes(network);
    network.layers = network.nodesPerLayer.size();
    intBiasforImport(network);
    intWeights(network);
    

        if (line == "{") {
            while (line != "(") {
                getline(file, line);
                if (line == "}") {
                    currentLayer++;
                }
                else if (line == ";") {
                    currentNode++;
                }
                else if (line == ";}") {

                    currentLayer++;
                    currentNode = 0;
                    getline(file, line);
                    
                }
                else {
                    network.weights[currentLayer][currentNode].push_back(std::stod(line));
                }

            }
                while (true) {
                    getline(file, line);
                     if (line == ";") {
                        currentBiasLayer++;
                    }
                    else if (line == ";)") {
                        break;
                    }
                    else {
                        network.bias[currentBiasLayer].push_back(stod(line));
                    }
                       
                    
                
            }
        }

}

double xavierInitialization(int fan_in, int fan_out) {
    // Use Xavier initialization (Glorot initialization)
    auto start = std::chrono::high_resolution_clock::now();
    double variance = 1.0 / (fan_in + fan_out);
    double stddev = sqrt(variance);

    // Generate random values with mean 0 and the calculated standard deviation
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0, stddev);
    high_resolution_clock::time_point startTime = high_resolution_clock::now();
    /*while ((std::chrono::high_resolution_clock::now() - start).count() < 1) {
        int i = 1;
        i = 0;
    }*/
    return distribution(generator);
}
double sigmoid(double m1) {
    return 1 / (1 + exp(-m1));
}
double sigmoidDerivative(double m1) {
    return exp(m1) / ((exp(m1) + 1) * (exp(m1) + 1));
}
double calcGradient(double preActivation, double EXresult, int weightLayer, Network& network) {
    double activated = sigmoid(preActivation);
    weightLayer = weightLayer - (network.weights.size() - 1);
    return preActivation* sigmoidDerivative(preActivation)* (2*(activated - EXresult));
}

double findLargest(std::vector<double> vec) {
    if (vec.empty()) {
        // Handle the case when the vector is empty, or you can return an error code.
        return -1; // Return -1 to indicate an error.
    }

    double maxVal = vec[0];
    int position = 0;

    for (int i = 1; i < vec.size(); i++) {
        if (vec[i] > maxVal) {
            maxVal = vec[i];
            position = i;
        }
    }

    return position;
}

