cmake_minimum_required(VERSION 3.10.0)
project(image_evaluator VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(../../include)
file(GLOB source_files src/*.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})