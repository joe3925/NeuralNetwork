
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <algorithm>
#include <cuchar>
#include <string>
#include <cstring>


constexpr double Finaltarget = 1;
constexpr double learningRate = 0.5;

constexpr int firstLayerNamount = 784;
constexpr int  secondLayerNamount = 16;
constexpr int  thirdLayerNamount = 10;


double inputValue[784];
double weightValue = 0;

int numberOfLables;

std::vector<std::vector<double>> images;
std::vector<int> labels;

std::vector<double> firstLayerN;
std::vector<double> secondLayerN;
std::vector<double> thirdLayerN;
std::vector<double> firstLayerW;
std::vector<double> secondLayerW;
std::vector<double> firstLayerWeighted;
std::vector<double> secondLayerWeighted;

const std::string mnistImagesFile = "..\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "..\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";

double sigmoid (double m1);
double cost(double target, double result);
double fixWeight(double initalValue, double outputValue);
double xavierInitialization(int fan_in, int fan_out);
void readMNISTData(std::vector<std::vector<double>>& images, std::vector<int>& labels);
double sumVector(std::vector<double> vector,int startAmount, int endAmount);
void TestNetwork(std::vector<double> image, int label, double& costReturn, int& result);


int main()
{
    for (int i = 0; i < secondLayerNamount * firstLayerNamount; i++) {
        firstLayerW.push_back(xavierInitialization(firstLayerNamount, secondLayerNamount));
    }
    for (int i = 0; i < secondLayerNamount * thirdLayerNamount; i++) {
        secondLayerW.push_back(xavierInitialization(secondLayerNamount, thirdLayerNamount));
    }
    double cost;
    int result;
    readMNISTData(images, labels);
    TestNetwork(images[0], labels[0], cost, result);
    std::cout << cost << "\n";
    std::cout << result << "\n";
    return 0;
}
double sigmoid (double m1) {
    return 1 / (1 + exp(-m1));
}
double cost(double target, double result)
{
    return ((target - result) * (target - result));
}
double fixWeight(double initalValue, double outputValue)
{
    return initalValue + (learningRate* 2 *(Finaltarget- outputValue));
}
double xavierInitialization(int fan_in, int fan_out) {
    // Use Xavier initialization (Glorot initialization)
    double variance = 1.0 / (fan_in + fan_out);
    double stddev = sqrt(variance);

    // Generate random values with mean 0 and the calculated standard deviation
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0, stddev);
    return distribution(generator);
}
double sumVector(std::vector<double> vector,int startAmount, int endAmount) {
    int i = startAmount;
    double total = 0;
    while (i < endAmount) {
        total = total + vector[i];
        i++;
    }
    return total;
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

void TestNetwork(std::vector<double> image, int label, double &costReturn, int &result) {
    firstLayerN = image;



    for (int i = 0; i < secondLayerNamount; i++) {
        int j = 0;

        firstLayerWeighted.push_back(sumVector(firstLayerW, firstLayerNamount * i, firstLayerNamount * (i + 1)));


        secondLayerN.push_back(sigmoid(firstLayerWeighted[i] / secondLayerNamount));
    }




    for (int i = 0; i < thirdLayerNamount; i++) {
        int j = 0;

        secondLayerWeighted.push_back(sumVector(secondLayerW, secondLayerNamount * i, secondLayerNamount * (i + 1)));


        thirdLayerN.push_back(sigmoid(secondLayerWeighted[i] / thirdLayerNamount));
    }
    result = findLargest(thirdLayerN);
    costReturn = cost(label, result);
}




// MNIST dataset constants
const int numImages = 60000;
const int imageSize = 28 * 28;

// Read MNIST images and labels
void readMNISTData(std::vector<std::vector<double>>& images, std::vector<int>& labels) {
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
        images.push_back(image);
    }

    // Read the MNIST label file header
    char buffer1[8];
    labelStream.read(buffer1, 8);

    // Read the labels
    for (int i = 0; i < numImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }

    // Close the streams when done
    imageStream.close();
    labelStream.close();
}


