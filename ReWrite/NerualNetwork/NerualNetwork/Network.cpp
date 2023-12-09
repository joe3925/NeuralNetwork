#include "Network.h"
#include <iostream>
#include <fstream>
#include <windows.h>

auto images = std::make_unique<std::vector<std::vector<double>>>(60000, std::vector<double>(784));
auto labels = std::make_unique<std::vector<int>>(60000);
std::vector<std::vector<double>> testImages;
std::vector<int> testLabels;
const double learningRate = 0.02;
const std::string mnistImagesFile = "..\\..\\NerualNetwork\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
const std::string mnistTestImagesFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-images.idx3-ubyte";
const std::string mnistTestLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-labels.idx1-ubyte";

bool testData = true;
double correct;
double wrong;
std::vector<int> label1 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
std::vector<int> networkArchitecture = { 784,10 };
void readMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath);
void readTestMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath);
void printNumber(std::vector<double> num);
double relu(double x);
double relu_derivative(double x);
int main()
{
    //read data into memory and create the network
    readMNISTData(*images, *labels, mnistImagesFile, mnistLabelsFile);
    //intitate our network
    Network network(images, labels, networkArchitecture, sigmoid, sigmoidDerivative);
    intNodes(network);
    printNumber(network.images[3245]);
    images->clear();
    labels->clear();
    
    if (xavierIntWeights(network))
    {
        return 1;
    }
    if (intBias(network))
    {
        return 1;
    }
    for (int i = 0; i < 60000; i++) {
        label1 = { 0,0,0,0,0,0,0,0,0,0 };
        label1[network.labels[i]] = 1;
        network.currentTargetOutput = label1;
        feedForward(network, i);
        //std::vector<double> grad = calcGradient( label1[0], 2, network);
        //grad;
        double total = 0;
        for (int j = 0; j < network.nodesPerLayer[network.nodesPerLayer.size() - 1]; j++) {
            total = total + (network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1][j] - label1[j]) * (network.layersValuesPostActivation[network.layersValuesPostActivation.size() - 1][j] - label1[j]);
        }
        std::cout << total / network.nodesPerLayer[network.nodesPerLayer.size() - 1] << "\n";
        std::cout << i << "\n";
        if (network.results[i] == network.labels[i]) {
            std::cout << "CORRECT" << "\n";
            correct++;
        }
        else {
            std::cout << "WRONG ):" << "\n";
            wrong++;

        }        
        backPropagate(network, learningRate);
        std::cout << ((correct / (i + 1)) * 100) << "\n";

        std::cout << "\n";
    }
    exportNetwork("MNIST.net",network);
    if (testData == true) {
        correct = 0;
        wrong = 0;
        network.results.clear();
        readTestMNISTData(testImages, testLabels, mnistTestImagesFile, mnistTestLabelsFile);
        network.labels = testLabels;
        network.images = testImages;
        for (int i = 0; i < 10000; i++) {
            label1 = { 0,0,0,0,0,0,0,0,0,0 };
            label1[network.labels[i]] = 1;
            feedForward(network, i);
            //std::vector<double> grad = calcGradient( label1[0], 2, network);
            //grad;            
            std::cout << i << "\n";
            if (network.results[i] == network.labels[i]) {
                std::cout << "CORRECT" << "\n";
                correct++;
            }
            else {
                std::cout << "WRONG ):" << "\n";
                wrong++;

            }
            std::cout << ((correct / (i + 1)) * 100) << "\n";

            std::cout << "\n";
        }
    }

    
    return 0;
    
}






















const int numImages = 60000;
const int imageSize = 28 * 28;
const int numTestImages = 10000;

//Possible to maybe read and realease the image files one by one but im not sure of the potential preformance hit on the network so ill just preload them 
// Read MNIST images and labels
void readMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath) {
    // Open the binary image file
    std::ifstream imageStream(imagePath, std::ios::binary);
    if (!imageStream) {
        std::cerr << "Failed to open image file." << std::endl;
        return;
    }

    // Open the binary label file
    std::ifstream labelStream(labelPath, std::ios::binary);
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
        images1[i]=image;
    }

    // Read the MNIST label file header
    char buffer1[8];
    labelStream.read(buffer1, 8);

    // Read the labels
    for (int i = 0; i < numImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        labels1[i] = static_cast<int>(label);
    }

    // Close the streams when done
    imageStream.close();
    labelStream.close();
}
void readTestMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath) {
    // Open the binary image file
    std::ifstream imageStream(imagePath, std::ios::binary);
    if (!imageStream) {
        std::cerr << "Failed to open image file." << std::endl;
        return;
    }

    // Open the binary label file
    std::ifstream labelStream(labelPath, std::ios::binary);
    if (!labelStream) {
        std::cerr << "Failed to open label file." << std::endl;
        return;
    }

    char buffer[16];
    imageStream.read(buffer, 16);

    // Read the images
    for (int i = 0; i < numTestImages; i++) {
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
    for (int i = 0; i < numTestImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        labels1.push_back(static_cast<int>(label));
    }

    // Close the streams when done
    imageStream.close();
    labelStream.close();
}

void printNumber(std::vector<double> num) {
    std::cout << "\n";
        for (int z = 0; z < 28; z++ ) {
            for (int i = 28 * z; i < 28 * (z + 1); i++) {
                if (num[i] != 0) {
                    std::cout << "&";
                }
                else {
                    std::cout << "-";
                }
            }
           std::cout << "\n";
        }
}

double relu(double x) {
    return ((x*-1) > 0) ? x : 0;
}

// Derivative of ReLU Function
double relu_derivative(double x) {
    return ((x*-1) > 0) ? 1 : 0;
}
