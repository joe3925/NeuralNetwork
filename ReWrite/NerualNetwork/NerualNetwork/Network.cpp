#include "Network.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

const int numImages = 60000;
const int imageSize = 28 * 28;
const int numTestImages = 10000;
const int VALIDATION_INTERVAL = 10000;
const int VALIDATION_SET_SIZE = 5000;
const int epochs = 2500;
double learningRate = 0.00017;

const std::string mnistImagesFile = "../../NerualNetwork/TrainingData/train-images-idx3-ubyte/train-images.idx3-ubyte";
const std::string mnistLabelsFile = "../../NerualNetwork/TrainingData/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
const std::string mnistTestImagesFile = "../../NerualNetwork/TrainingData/t10k-images.idx3-ubyte";
const std::string mnistTestLabelsFile = "../../NerualNetwork/TrainingData/t10k-labels.idx1-ubyte";

auto* images = new std::vector<std::vector<double>>(numImages, std::vector<double>(imageSize));
auto* labels = new std::vector<int>(numImages);
auto* testImages = new std::vector<std::vector<double>>(numTestImages, std::vector<double>(imageSize));
auto* testLabels = new std::vector<int>(numTestImages);

std::vector<int> networkArchitecture = { 784,64, 10 };

void readMNISTData(std::vector<std::vector<double>>* images, std::vector<int>* labels, const std::string& imagePath, const std::string& labelPath);
void readTestMNISTData(std::vector<std::vector<double>>* images, std::vector<int>* labels, const std::string& imagePath, const std::string& labelPath);
void printNumber(const std::vector<double>& num);
double relu(double x);
double relu_derivative(double x);
static double sigmoid(double m1)
{
    return 1 / (1 + exp(-m1));
}

static double sigmoidDerivative(double m1)
{
    double solve = exp(m1) / ((exp(m1) + 1) * (exp(m1) + 1));
    return solve;
}
int main() {
    // Vectors to hold validation images, labels, cost, training data, and accuracy per epoch
    std::vector<std::vector<double>> validationImages;
    std::vector<int> validationLabels;
    std::vector<double> cost;
    std::vector<double> trainData;
    std::vector<double> EpochAcc;
    std::vector<double> val;

    // Read MNIST data and split into training and validation sets
    readMNISTData(images, labels, mnistImagesFile, mnistLabelsFile);
    readTestMNISTData(testImages, testLabels, mnistTestImagesFile, mnistTestLabelsFile);
    validationImages.assign(testImages->begin(), testImages->begin() + VALIDATION_SET_SIZE);
    validationLabels.assign(testLabels->begin(), testLabels->begin() + VALIDATION_SET_SIZE);
    testImages->erase(testImages->begin(), testImages->begin() + VALIDATION_SET_SIZE);
    testLabels->erase(testLabels->begin(), testLabels->begin() + VALIDATION_SET_SIZE);

    Network network(networkArchitecture, sigmoid, sigmoidDerivative);
    network.intNodes();

    // Initialize weights and biases using Xavier initialization
    if (network.xavierIntWeights() || network.intBias()) {
        return 1;
    }

    // Training loop for a specified number of epochs
    for (int j = 0; j < epochs; j++) {
        double correct = 0;
        // Iterate through all training images
        for (int i = 0; i < numImages; i++) {
            std::vector<int> label1(10, 0);
            label1[(*labels)[i]] = 1; // Create one-hot encoded label

            // Validate the network at specified intervals
            if ((i + 1) % VALIDATION_INTERVAL == 0) {
                double validationCorrect = 0;
                double mse = 0.0;
                // Iterate through validation images
                for (int v = 0; v < validationImages.size(); v++) {
                    std::vector<int> validationLabel1(10, 0);
                    validationLabel1[validationLabels[v]] = 1;
                    network.setCurrentTargetOutput(validationLabel1);

                    int predictedLabel = network.feedForward(validationImages[v], false);
                    if (predictedLabel == validationLabels[v]) {
                        validationCorrect++;
                    }

                    // Calculate error for each output node
                    for (int k = 0; k < network.getLayersValuesPostActivation().back().size(); k++) {
                        double error = validationLabel1[k] - network.getLayersValuesPostActivation().back()[k];
                        mse += error * error;
                    }
                }

                mse /= validationImages.size();
                cost.push_back(mse);

                double validationAccuracy = (validationCorrect / validationImages.size()) * 100.0;
                std::cout << validationAccuracy << "\n";
                val.push_back(validationAccuracy);

                // Stop training if validation accuracy reaches 93%
                if (validationAccuracy >= 93) {
                    goto test;
                }

            }

            
            // Train the network on the current image

            network.setCurrentTargetOutput(label1);

            network.feedForward((*images)[i], true);

            if (network.results[i + j * numImages] == (*labels)[i]) {
                correct++;
            }
            network.backPropagate(learningRate);
            std::chrono::high_resolution_clock::time_point loopend = std::chrono::high_resolution_clock::now();
        }
        // Store epoch accuracy and training data
        EpochAcc.push_back(val.back());
        trainData.push_back((correct / numImages) * 100);
        std::cout << j << "\n";
    }

test:
    // Save the trained network to a file
    network.exportNetwork("MNIST.net");

    // Test the network on test data if enabled
    if (true) {
        std::vector<double> testData;
        double correct = 0;
        network.results.clear();
        readTestMNISTData(testImages, testLabels, mnistTestImagesFile, mnistTestLabelsFile);

        // Iterate through test images
        for (int i = 0; i < numTestImages; i++) {
            std::vector<int> label1(10, 0);
            label1[(*testLabels)[i]] = 1;
            network.feedForward((*testImages)[i], true);

            if (network.results[i] == (*testLabels)[i]) {
                correct++;
            }
            if (i % (numTestImages / 240) == 0) {
                testData.push_back((correct / (i + 1)) * 100);
            }
        }
    }
    images->clear();
    labels->clear();
    testImages->clear();
    testLabels->clear();
    delete images;
    delete labels;
    delete testImages;
    delete testLabels;
    std::vector<int> logScaleCost(cost.size());
    for (int i = 0; i < cost.size(); ++i) {
        logScaleCost[i] = i + 1;
    }
    std::vector<int> logScaleEpoch(EpochAcc.size());
    for (int i = 0; i < EpochAcc.size(); ++i) {
        logScaleEpoch[i] = i + 1;
    }

    plt::title("Training Accuracy vs. Epochs");
    plt::xlabel("Epochs");
    plt::ylabel("Training Accuracy (%)");
    plt::grid(true);
    plt::semilogx(logScaleEpoch, trainData, "r-");
    plt::save("training_accuracy.png");

    // Plot Validation Accuracy vs. Epochs
    plt::figure();
    plt::title("Validation Accuracy vs. Epochs");
    plt::xlabel("Epochs");
    plt::ylabel("Validation Accuracy (%)");
    plt::grid(true);
    plt::semilogx(logScaleEpoch, EpochAcc);
    plt::save("validation_accuracy.png");

    // Plot Cost vs. Epochs if cost data is available
    if (!cost.empty()) {
        plt::figure();
        plt::title("Cost vs. Epochs");
        plt::xlabel("Epochs");
        plt::ylabel("Cost");
        plt::grid(true);

        // Create a vector for the x-axis (epochs)


        // Use semilogy to plot the cost on a log scale
        plt::semilogx(logScaleCost, cost);

        plt::save("cost_vs_epochs.png");
    }

    plt::show();
    

    return 0;
}
void readMNISTData(std::vector<std::vector<double>>* images, std::vector<int>* labels, const std::string& imagePath, const std::string& labelPath) {
    std::ifstream imageStream(imagePath, std::ios::binary);
    if (!imageStream) {
        std::cerr << "Failed to open image file." << std::endl;
        return;
    }

    std::ifstream labelStream(labelPath, std::ios::binary);
    if (!labelStream) {
        std::cerr << "Failed to open label file." << std::endl;
        return;
    }

    char buffer[16];
    imageStream.read(buffer, 16);

    for (int i = 0; i < numImages; i++) {
        std::vector<double> image(imageSize, 0.0);
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            imageStream.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<double>(pixel) / 255.0;
        }
        (*images)[i] = image;
    }

    char buffer1[8];
    labelStream.read(buffer1, 8);

    for (int i = 0; i < numImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        (*labels)[i] = static_cast<int>(label);
    }

    imageStream.close();
    labelStream.close();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(images->begin(), images->end(), std::default_random_engine(seed));
    std::shuffle(labels->begin(), labels->end(), std::default_random_engine(seed));
}

void readTestMNISTData(std::vector<std::vector<double>>* images, std::vector<int>* labels, const std::string& imagePath, const std::string& labelPath) {
    std::ifstream imageStream(imagePath, std::ios::binary);
    if (!imageStream) {
        std::cerr << "Failed to open image file." << std::endl;
        return;
    }

    std::ifstream labelStream(labelPath, std::ios::binary);
    if (!labelStream) {
        std::cerr << "Failed to open label file." << std::endl;
        return;
    }

    char buffer[16];
    imageStream.read(buffer, 16);

    for (int i = 0; i < numTestImages; i++) {
        std::vector<double> image(imageSize, 0.0);
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            imageStream.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<double>(pixel) / 255.0;
        }
        (*images)[i] = image;
    }

    char buffer1[8];
    labelStream.read(buffer1, 8);

    for (int i = 0; i < numTestImages; i++) {
        unsigned char label;
        labelStream.read(reinterpret_cast<char*>(&label), 1);
        (*labels)[i] = static_cast<int>(label);
    }

    imageStream.close();
    labelStream.close();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(images->begin(), images->end(), std::default_random_engine(seed));
    std::shuffle(labels->begin(), labels->end(), std::default_random_engine(seed));
}

void printNumber(const std::vector<double>& num) {
    std::cout << "\n";
    for (int z = 0; z < 28; z++) {
        for (int i = 28 * z; i < 28 * (z + 1); i++) {
            if (num[i] != 0) {
                std::cout << "&";
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
    return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}