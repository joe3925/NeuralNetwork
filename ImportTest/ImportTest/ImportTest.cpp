// ImportTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Network.hpp"

const std::string mnistTestImagesFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-images.idx3-ubyte";
const std::string mnistTestLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-labels.idx1-ubyte";
const std::string networkPath = "..\\..\\ReWrite\\NerualNetwork\\NerualNetwork\\MNIST.net";
const int imageSize = 28 * 28;
const int numTestImages = 10000;
void readTestMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath);
std::vector<std::vector<double>> images;
std::vector<int> labels;
double correct = 0;

int main()
{
    readTestMNISTData(images, labels, mnistTestImagesFile, mnistTestLabelsFile);
    Network network(images, labels, {{NULL}});
    importNetwork(networkPath,network);
    for (int i = 0; i < network.labels.size(); i++) {
        feedForward(network, i);
        std::cout << network.results[i] << "\n";
        std::cout << network.labels[i] << "\n";
        if (network.results[i] == network.labels[i]) {
            std::cout << "correct" << "\n";
            correct++;
        }
        else {
            std::cout << "wrong" << "\n";
        }
        std::cout << (correct / (i + 1)) * 100 << "\n";
        std::cout << "\n";
    }
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
