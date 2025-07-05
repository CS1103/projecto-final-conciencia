cmake_minimum_required(VERSION 3.10.0)
project(batch_evaluator VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenMP REQUIRED)

include_directories(../../include)
file(GLOB source_files src/*.cpp)

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)