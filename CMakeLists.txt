cmake_minimum_required(VERSION 3.12)
project(pong_ai)

set(CMAKE_CXX_STANDARD 20)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

add_subdirectory(generator)
add_subdirectory(tests)

include_directories(./include)
file(GLOB source_files src/*.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
