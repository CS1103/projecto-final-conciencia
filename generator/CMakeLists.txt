cmake_minimum_required(VERSION 3.10.0)
project(popn_generator VERSION 0.1.0 LANGUAGES C CXX)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(popn_generator main.cpp)
target_link_libraries( popn_generator ${OpenCV_LIBS} )

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/assets" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")