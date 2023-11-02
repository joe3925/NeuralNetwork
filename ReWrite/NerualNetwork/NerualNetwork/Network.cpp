#include "Network.h"
#include <iostream>
#include <fstream>

Network network;
std::vector<std::vector<double>> images;
std::vector<int> labels;
const std::string mnistImagesFile = "C:\\Users\\Boden\\Documents\\NeuralNetwork\\ReWrite\\NerualNetwork\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "C:\\Users\\Boden\\Documents\\NeuralNetwork\\ReWrite\\NerualNetwork\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
void readMNISTData(std::vector<std::vector<double>>&images1, std::vector<int>&labels1);
int main()
{
    //read data into memory and create the network
    readMNISTData(images, labels);
    if (network.intNetworkObject(images,labels, {784,16,10},network ))
    {
        std::cerr<<"unknown error with network initialization";
        return 1;
    }
    if (network.xavierIntWeights(network))
    {
        std::cerr<<"Layers, nodesPerLayer or both were initialized wrong";
        return 1;
    }
    if (network.intBias(network))
    {
        std::cerr<<"Layers, nodesPerLayer or both were initialized wrong";
        return 1;
    }
    
    network.feedForward(network, 0);
    images.clear();
    labels.clear();
    
    return 0;
    
}






















const int numImages = 60000;
const int imageSize = 28 * 28;

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