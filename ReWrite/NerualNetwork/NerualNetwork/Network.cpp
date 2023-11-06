#include "Network.h"
#include <iostream>
#include <fstream>

Network network;
std::vector<std::vector<double>> images;
std::vector<int> labels;
const std::string mnistImagesFile = "..\\..\\NerualNetwork\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
std::vector<int> label1 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
std::vector<int> networkArchitecture = { 784,16,10 };
void readMNISTData(std::vector<std::vector<double>>&images1, std::vector<int>&labels1);
int main()
{

    //read data into memory and create the network
    readMNISTData(images, labels);
    //intitate our network
    network.intNetworkObject(images, labels, networkArchitecture, network);
    images.clear();
    labels.clear();
    
    if (network.xavierIntWeights(network))
    {
        return 1;
    }
    if (network.intBias(network))
    {
        return 1;
    }
    network.exportNetwork("network.net", network);
    for (int i = 0; i < network.images.size(); i++) {
        network.feedForward(network, i); 
        double total = 0;
        label1 = { 0,0,0,0,0,0,0,0,0,0 };
        label1[network.labels[i]] = 1;
        for (int j = 0; j < network.nodesPerLayer[network.nodesPerLayer.size() - 1]; j++) {
            total = total + (network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1][j] - label1[j]) * (network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1][j] - label1[j]);
        }
        
        std::cout << total / network.nodesPerLayer[network.nodesPerLayer.size() - 1] << "\n";
        std::cout << i << "\n";


    }

    
    return 0;
    
}






















const int numImages = 60000;
const int imageSize = 28 * 28;
//Possible to maybe read and realease the image files one by one but im not sure of the potential preformance hit on the network so ill just preload them 
// Read MNIST images and labels
void readMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1) {
    // Open the binary image file
    std::ifstream imageStream(mnistImagesFile, std::ios::binary);
    if (!imageStream) {
        std::cerr << "Failed to open image file." << std::endl;
        return;
    }

    // Open the binary label file
    std::ifstream labelStream(mnistLabelsFile, std::ios::binary);
    if (!labelStream) {
        std::cerr << "Failed to open label file." << std::endl;
        return;
    }

    char buffer[16];
    imageStream.read(buffer, 16);

    // Read the images
    for (int i = 0; i < numImages; i++) {
        std::vector<double> image(imageSize, 0.0);
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            imageStream.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<double>(pixel) / 255.0;  // Normalize pixel values to [0, 1]
        }
        images1.push_back(image);
    }

    // Read the MNIST label file header
    char buffer1[8];
    labelStream.read(buffer1, 8);

    // Read the labels
    for (int i = 0; i < numImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        labels1.push_back(static_cast<int>(label));
    }

    // Close the streams when done
    imageStream.close();
    labelStream.close();
}