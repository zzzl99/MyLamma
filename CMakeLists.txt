cmake_minimum_required(VERSION 3.16)
project(llama_infer)

find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)
add_subdirectory(test)
