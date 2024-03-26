#include "Network.h"
#include <iostream>
#include <fstream>
#include <windows.h>
#include <threadpoolapiset.h>
#include "matplotlibcpp.h"

auto images = std::vector<std::vector<double>>(60000, std::vector<double>(784));
auto labels = std::vector<int>(60000);
std::vector<std::vector<double>> testImages;
std::vector<int> testLabels;
std::vector<bool> sample;
double learningRate = 0.005;
const std::string mnistImagesFile = "..\\..\\NerualNetwork\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
const std::string mnistTestImagesFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-images.idx3-ubyte";
const std::string mnistTestLabelsFile = "..\\..\\NerualNetwork\\TrainingData\\t10k-labels.idx1-ubyte";
const int VALIDATION_INTERVAL = 5000;
const int VALIDATION_SET_SIZE = 10000;
const int epochs = 10;
std::vector<double> val;


bool testData = true;

std::vector<int> label1 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
std::vector<int> networkArchitecture = { 784,10 };
void readMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath);
void readTestMNISTData(std::vector<std::vector<double>>& images1, std::vector<int>& labels1, std::string imagePath, std::string labelPath);
void printNumber(std::vector<double> num);
double relu(double x);
double relu_derivative(double x);
double getPercentOfVector(std::vector<bool> par1);
namespace plt = matplotlibcpp;

int main()
{

    

    std::vector<std::vector<double>> validationImages;
    std::vector<int> validationLabels;
    std::vector<double> cost;
    std::vector<double> trainData;
    std::vector<double> EpochAcc;



    //read data into memory and create the network
    readMNISTData(images, labels, mnistImagesFile, mnistLabelsFile);
    readTestMNISTData(testImages, testLabels, mnistTestImagesFile, mnistTestLabelsFile);
    validationImages.assign(testImages.begin(), testImages.begin() + VALIDATION_SET_SIZE);
    validationLabels.assign(testLabels.begin(), testLabels.begin() + VALIDATION_SET_SIZE);
    testImages.erase(testImages.begin(), testImages.begin() + VALIDATION_SET_SIZE);
    testLabels.erase(testLabels.begin(), testLabels.begin() + VALIDATION_SET_SIZE);
    //intitate our network
    Network network(networkArchitecture, sigmoid, sigmoidDerivative);
    intNodes(network);
    printNumber(images[45368]);


    if (xavierIntWeights(network))
    {
        return 1;
    }
    if (intBias(network))
    {
        return 1;
    }



    for (int j = 0; j < epochs; j++)
    {
        double recentCorrect = 0;
        double recentWrong = 0;
        double sumSquaredErrors = 0;
        double correct = 0;
        double wrong = 0;

        for (int i = 0; i < 60000; i++) {
            label1 = { 0,0,0,0,0,0,0,0,0,0 };
            label1[labels[i]] = 1;
            correct = 0;
            wrong = 0;

            //validate network
            if ((i + 1) % VALIDATION_INTERVAL == 0)
            {
                double validationCorrect = 0;
                double MSE = 0;
                for (int v = 0; v < validationImages.size(); v++) {
                    std::vector<int> validationLabel1 = { 0,0,0,0,0,0,0,0,0,0 };
                    validationLabel1[validationLabels[v]] = 1;
                    network.currentTargetOutput = validationLabel1;

                    // Check if the prediction is correct

                    int predictedLabel = feedForward(network, validationImages[v], false);
                    if (predictedLabel == validationLabels[v]) {
                        validationCorrect++;
                    }
                    for (int z = 0; z < network.nodesPerLayer[network.nodesPerLayer.size() - 1]; z++) {
                        MSE += (network.layersValuesPostActivation[network.nodesPerLayer.size() - 1][z] - validationLabels[z]) * (network.layersValuesPostActivation[network.nodesPerLayer.size() - 1][z] - validationLabels[z]);
                    }
                    cost.push_back(MSE / network.nodesPerLayer[network.nodesPerLayer.size() - 1]);
                }
                double validationAccuracy = (validationCorrect / validationImages.size()) * 100.0;
                std::cout << validationAccuracy << "\n";
                val.push_back(validationAccuracy);
                
                if (validationAccuracy >= 93) {
                    goto test;
                   // return 0;
                }

            }
            //


            network.currentTargetOutput = label1;
            feedForward(network, images[i], true);

            if (network.results[i] == labels[i]) {
                correct++;


            }
            else {
                wrong++;

            }



            backPropagate(network, learningRate);
        }
        EpochAcc.push_back(val[val.size() - 1]);
        trainData.push_back(((correct / (60000 + 1)) * 100));
        std::cout << j << "\n";
    }
    test:
    exportNetwork("MNIST.net",network);
    if (testData == true) {
        std::vector<double> testData;
        double correct = 0;
        double wrong = 0;
        network.results.clear();
        readTestMNISTData(testImages, testLabels, mnistTestImagesFile, mnistTestLabelsFile);

        for (int i = 0; i < 10000; i++) {
            label1 = { 0,0,0,0,0,0,0,0,0,0 };
            label1[testLabels[i]] = 1;
            feedForward(network, testImages[i], true);
            //std::vector<double> grad = calcGradient( label1[0], 2, network);
            //grad;            
            std::cout << i << "\n";
            if (network.results[i] == testLabels[i]) {
                std::cout << "CORRECT" << "\n";
                correct++;
            }
            else {
                std::cout << "WRONG ):" << "\n";
                wrong++;

            }
            std::cout << ((correct / (i + 1)) * 100) << "\n";
            if (i % (10000 / 240) == 0) {
                testData.push_back(((correct / (i + 1)) * 100));
            }

            std::cout << "\n";

        }

        plt::plot(val);
        plt::plot(testData);
        plt::figure();
        plt::plot(trainData);
        plt::plot(EpochAcc);
        plt::figure();
        plt::plot(cost);
        plt::show();
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
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(images1.begin(), images1.end(), std::default_random_engine(seed));
    shuffle(labels1.begin(), labels1.end(), std::default_random_engine(seed));

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
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(images1.begin(), images1.end(), std::default_random_engine(seed));
    shuffle(labels1.begin(), labels1.end(), std::default_random_engine(seed));
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

double getPercentOfVector(std::vector<bool> par1) {
    double correct = 0;
    double wrong = 0;

    for (int i = 0; i<par1.size(); i++) {
        if (par1[i]) {
            correct++;
        }
        else {
            wrong++;
        }
    }
    return (correct/par1.size()) * 100;
}

double relu(double x) {
    return ((x*-1) > 0) ? x : 0;
}

// Derivative of ReLU Function
double relu_derivative(double x) {
    if (x >= 0) {
        return 1;
    }
    else {
        return 0;
    }
}
