cmake_minimum_required(VERSION 3.26)
project(ReWrite)

set(CMAKE_CXX_STANDARD 14)

include_directories(NerualNetwork/NerualNetwork)

add_executable(ReWrite
        NerualNetwork/NerualNetwork/Network.cpp
        NerualNetwork/NerualNetwork/Network.h)
