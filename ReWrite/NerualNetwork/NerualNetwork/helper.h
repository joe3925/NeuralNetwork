#pragma once
#include <vector>

template <typename T>
static int findLargest(const std::vector<T>& vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

template <typename T>
int findSmallestPosition(const std::vector<T>& vec) {
    if (vec.empty()) {
        // Handle the case where the vector is empty
        std::cout << "Vector is empty." << std::endl;
        return -1; // Return an invalid position
    }

    int smallestPosition = 0;
    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[smallestPosition]) {
            smallestPosition = i;
        }
    }

    return smallestPosition;
}
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            \
        } \
    } while (0)

#define CUDA_CHECK_AND_FAULT(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(0);\
        } \
    } while (0)
