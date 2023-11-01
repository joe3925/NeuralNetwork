
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <algorithm>
#include <cuchar>
#include <string>
#include <cstring>
#include <thread>
#include <windows.h>
#include <chrono>


constexpr double Finaltarget = 1;
constexpr double learningRate = 0.1;

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
std::vector<double> fisrtLayerB;
std::vector<double> secondLayerB;
std::vector<double> firstLayerWeighted;
std::vector<double> secondLayerWeighted;
std::vector<double>  SecondLayerError;
std::vector<double>  FirstLayerError;

std::vector<int> label1 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


const std::string mnistImagesFile = "..\\TrainingData\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
const std::string mnistLabelsFile = "..\\TrainingData\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";

void readMNISTData(std::vector<std::vector<double>>& images, std::vector<int>& labels);
void TestNetwork(std::vector<double> image, std::vector<int> label, double& costReturn, double& result);
void AdjustWeights(std::vector<double>& weights, std::vector<double>& errors, std::vector<double>& outputs, double learningRate);
double sigmoid (double m1);
double cost(std::vector <double> target, std::vector<double> result);
double xavierInitialization(int fan_in, int fan_out);
double sumVector(std::vector<double> vector, int startAmount, int endAmount);
std::vector<double> derivativeOfFinalCost(std::vector<double> input, std::vector<int> expected);
std::vector<double> reverseVector(std::vector<double> vec);
std::vector<double> multiplyVectorByVector(std::vector<double> vector, std::vector<double> vector1);
std::vector<double> multiplyVectorByMatrix(std::vector<double> vector, std::vector<double> matrix);
std::vector<double> applySigmoidToVector(const std::vector<double> input);
std::vector<double> applySigmoidDerivativeToVector(const std::vector<double> input);
std::vector<double> nodeErrorL1(std::vector<double> endError);
std::vector<double> nodeErrorL2(std::vector<int> expected);
std::vector<double> multiplyVectorByDouble(std::vector<double> input, double multiplier);
std::vector<double> weightVectorsL1(std::vector<double> error);
std::vector<double> weightVectorsL2(std::vector<double> error);
std::vector<int> generateRandomList(int minValue, int maxValue, int count);
void backpropagation(std::vector<double>& FirstLayerError, std::vector<double>& SecondLayerError);
double findLargest(std::vector<double> vec);

using namespace std::chrono;
void backpropagation(std::vector<double>& FirstLayerError, std::vector<double>& SecondLayerError) {
    // Update weights for the first layer
    for (int i = 0; i < secondLayerNamount; ++i) {
        for (int j = 0; j < firstLayerNamount; ++j) {
            int index = i * firstLayerNamount + j;
            firstLayerW[index] += learningRate * FirstLayerError[i] * firstLayerN[j];
        }
        fisrtLayerB[i] += learningRate * FirstLayerError[i];
    }

    // Update weights for the second layer
    for (int i = 0; i < thirdLayerNamount; ++i) {
        for (int j = 0; j < secondLayerNamount; ++j) {
            int index = i * secondLayerNamount + j;
            secondLayerW[index] += learningRate * SecondLayerError[i] * secondLayerN[j];
        }
        secondLayerB[i] += learningRate * SecondLayerError[i];
    }
}

int main()
{

    std::vector<int>datasetOrder = generateRandomList(0, 59999, 60000);
    //set starting weights 
    for (int i = 0; i < secondLayerNamount * firstLayerNamount; i++) {
        firstLayerW.push_back(xavierInitialization(firstLayerNamount, secondLayerNamount));
    }
    for (int i = 0; i < secondLayerNamount * thirdLayerNamount; i++) {
        secondLayerW.push_back(xavierInitialization(secondLayerNamount, thirdLayerNamount));
    }
    //set starting bias
    for (int i = 0; i < firstLayerNamount; i++) {
        fisrtLayerB.push_back(0);
    }    
    for (int i = 0; i < secondLayerNamount; i++) {
        secondLayerB.push_back(0);
    }
    double cost = 0.3;
    double result;
    double total = 0;
    readMNISTData(images, labels);
    int i = 0;
    while (cost > 0.2){
        while (i < images.size()) {
            label1[labels[i]] = 1;
            TestNetwork(images[i], label1, cost, result);
            total = total + cost;

            SecondLayerError = nodeErrorL2(label1);
            FirstLayerError = nodeErrorL1(SecondLayerError);
            fisrtLayerB = FirstLayerError;
             secondLayerB = SecondLayerError;
            firstLayerWeighted = weightVectorsL1(FirstLayerError);
            secondLayerWeighted = weightVectorsL2(SecondLayerError);
           // AdjustWeights(firstLayerW, FirstLayerError, secondLayerN, learningRate);
            //AdjustWeights(secondLayerW, SecondLayerError, thirdLayerN, learningRate);
           backpropagation(FirstLayerError, SecondLayerError);


            std::cout << cost << "\n";
            std::cout << "\n";
            std::cout << "iteration :" << i  << "{" << "\n";
            std::cout <<"correct answer: " << labels[i] << "\n";
            std::cout << "model answer : " << findLargest(thirdLayerN) << "\n";
            std::cout << "}";
            std::cout << "\n";

            std::cout << "\n";

            //system("cls");
            label1[labels[i]] = 0;
            i++;
            //AdjustWeights(secondLayerW, SecondWeightAdjustments);
            //AdjustWeights(firstLayerW, FirstWeightAdjustments);

            SecondLayerError.clear();
            FirstLayerError.clear();
            firstLayerWeighted.clear();
            secondLayerWeighted.clear();
            secondLayerN.clear();
            thirdLayerN.clear();



            total = total / images.size() / 6;
            //std::cout << total << "\n";
        }

    }
    total = total / images.size();
    std::cout << total << "\n";
    std::cout << cost << "\n";
    std::cout << result << "\n";
    return 0;
}
double sigmoid (double m1) {
    return 1 / (1 + exp(-m1));
}
//derivative of the activation function 
double sigmoidDerivative(double input) {
    return (exp(-input) / (((exp(-input) + 1) * exp(-input) + 1)));
}
double cost(std::vector <int> target, std::vector<double> result)
{
    int i = 0;
    double total = 0;
    while (i < result.size()) {
        total = total + (result[i] - target[i]) * (result[i] - target[i]);
        i++;
    }
    
    return total;
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}
//gradient descent is not fun
std::vector<double> nodeErrorL2(std::vector<int> expected)
{
    std::vector<double> preActivationN;
    double total = 0;
    int Nmultiplier = 0;
    while (Nmultiplier < thirdLayerNamount) {

        for (int i = 0; i < secondLayerNamount; i++) {
            total += secondLayerN[i] * secondLayerW[i + (thirdLayerNamount * Nmultiplier)] + secondLayerB[Nmultiplier];

        }
        preActivationN.push_back(total);
        Nmultiplier++;
        total = 0;
    }
        //just doing this to see the return easier in debugger 
        std::vector<double> final = multiplyVectorByVector(applySigmoidDerivativeToVector(preActivationN), derivativeOfFinalCost(thirdLayerN, expected));
       
        return final;
            
    

    
}
std::vector<double> nodeErrorL1 (std::vector<double> endError)
{
    std::vector<double> preActivationN;
    double total = 0;
    int Nmultiplier = 0;
    while (Nmultiplier < secondLayerNamount) {

        for (int i = 0; i < firstLayerNamount; i++) {
            total += firstLayerN[i] * firstLayerW[i + (secondLayerNamount * Nmultiplier)] + fisrtLayerB[Nmultiplier];

        }
        total = 0;
        preActivationN.push_back(total);
        Nmultiplier++;
    }
    return  multiplyVectorByVector(multiplyVectorByMatrix(endError, reverseVector(secondLayerW)), applySigmoidDerivativeToVector(preActivationN));
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
    while ((std::chrono::high_resolution_clock::now() - start).count() < 0) {
        int i = 1;
         i = 0;
    }
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
std::vector<double> weightVectorsL2(std::vector<double> error) {
    std::vector<double> result;
    for (int j = 0; j < secondLayerNamount; j++) {
        for (int i = 0; i < error.size(); i++) {
            result.push_back(error[i] * secondLayerN[j]);
        }
    }
    return result;
}
std::vector<double> weightVectorsL1(std::vector<double> error) {
    std::vector<double> result;
    for (int j = 0; j < firstLayerNamount; j++) {
        for (int i = 0; i < error.size(); i++) {
            result.push_back(error[i] * firstLayerN[j]);
        }
    }
    return result;
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
void AdjustWeights(std::vector<double>& weights, std::vector<double>& errors, std::vector<double>& outputs, double learningRate) {
    for (int j = 0; j < outputs.size(); j++) {
        for (int i = 0; i < weights.size(); ++i) {
            // Using the formula: weight = weight - learning_rate * error * output
            // where output is the output of the neuron connected to that weight
            weights[i] -= learningRate * errors[j] * sigmoid_derivative(outputs[j]);
        }
    }
}
void AdjustBias(std::vector<double>& Bias, std::vector <double> BiasAdjustments) {
    for (int i = 0; i < Bias.size(); i++) {
        Bias[i] = Bias[i] + BiasAdjustments[i];
    }
}
void TestNetwork(std::vector<double> image, std::vector<int> label, double &costReturn, double &result) {
    firstLayerN = image;
    double total = 0;
    int Nmultiplier = 0;
    while (Nmultiplier < secondLayerNamount) {
        //because of how I set everything up this multiplacation is needed (used list instead of a matrix)
        for (int i = 0; i < firstLayerNamount; i++) {
            total +=firstLayerN[i] * firstLayerW[i + (secondLayerNamount* Nmultiplier)] + fisrtLayerB[Nmultiplier];

        }   
        secondLayerN.push_back(sigmoid(total));
        total = 0;
        Nmultiplier++;
        
        
        
        /*firstLayerWeighted.push_back(sumVector(firstLayerW, firstLayerNamount * i, firstLayerNamount * (i + 1)));


        secondLayerN.push_back(sigmoid(firstLayerN[i] * (firstLayerWeighted[i] / secondLayerNamount)));*/
    }




    double total1 = 0;
    int Nmultiplier1 = 0;
    while (Nmultiplier1 < thirdLayerNamount) {
        //because of how I set everything up this multiplacation is needed (used list instead of a matrix)
        for (int i = 0; i < secondLayerNamount; i++) {
            total1 += secondLayerN[i] * secondLayerW[i + (thirdLayerNamount * Nmultiplier1)] + secondLayerB[Nmultiplier1];

        }
        thirdLayerN.push_back(1 - sigmoid(total1));
        total1 = 0;
        Nmultiplier1++;



        /*firstLayerWeighted.push_back(sumVector(firstLayerW, firstLayerNamount * i, firstLayerNamount * (i + 1)));


        secondLayerN.push_back(sigmoid(firstLayerN[i] * (firstLayerWeighted[i] / secondLayerNamount)));*/
    }

    result = thirdLayerN[findLargest(thirdLayerN)];
    costReturn = cost(label, thirdLayerN);

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


std::vector<int> generateRandomList(int minValue, int maxValue, int count) {
    if (count > (maxValue - minValue + 1) || count <= 0) {
        // Handle invalid input
        std::cout << "Invalid input parameters" << std::endl;
        return std::vector<int>();
    }

    std::vector<int> result;
    result.reserve(count);

    // Initialize an array with all possible values
    std::vector<int> allValues;
    for (int i = minValue; i <= maxValue; i++) {
        allValues.push_back(i);
    }

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < count; i++) {
        // Generate a random index in the range [0, size of remaining values)
        int randomIndex = std::rand() % allValues.size();

        // Add the value at the random index to the result list
        result.push_back(allValues[randomIndex]);

        // Remove the used value from the available values
        allValues.erase(allValues.begin() + randomIndex);
    }

    return result;
}


//I think this would transpose my vector as if it where a matrix 
std::vector<double> reverseVector(std::vector<double> vec) {
    int left = 0;
    int right = vec.size() - 1;

    while (left < right) {
        std::swap(vec[left], vec[right]);
        left++;
        right--;
    }
    return vec;
}
//element - wise 
std::vector<double> multiplyVectorByVector(std::vector<double> vector, std::vector<double> vector1) {
    std::vector<double> result;

    // Check if the input vectors have the same size
          
        for (size_t i = 0; i < vector.size(); ++i) {
            result.push_back(vector[i] * vector1[i]);
        }

        return result;
}

std::vector<double> multiplyVectorByMatrix( std::vector<double> vector, std::vector<double> matrix) {
    std::vector<double> result;
    std::vector<double> end;


    for (size_t i = 0; i < 16; ++i) {
        double sum = 0;
        for (size_t j = 0; j < vector.size(); ++j) {
            sum +=( vector[j] * matrix[j + (10 * i)]);
        }
        result.push_back(sum);
    }

    return result;
}
//vector and matrix stuff
std::vector<double> applySigmoidToVector(const std::vector<double> input) {
    std::vector<double> result;
    for (double value : input) {
        double sigmoidValue = sigmoid(value);
        result.push_back(sigmoidValue);
    }
    return result;
}
std::vector<double> applySigmoidDerivativeToVector(const std::vector<double> input) {
    std::vector<double> result;
    for (double value : input) {
        double sigmoidValue = sigmoidDerivative(value);
        result.push_back(sigmoidValue);
    }
    return result;
}

std::vector<double> derivativeOfFinalCost( std::vector<double> input, std::vector<int> expected) {
    std::vector<double> result;
    for (int i = 0; i < input.size(); i++) {
        double Value = 2 * (input[i] - expected[i]);
        result.push_back(Value);
    }
    return result;
}

std::vector<double> multiplyVectorByDouble( std::vector<double> input, double multiplier) {
    std::vector<double> result;
    for (double value : input) {
        result.push_back(value * multiplier);
    }
    return result;
}