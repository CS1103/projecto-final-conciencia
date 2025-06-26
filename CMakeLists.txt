cmake_minimum_required(VERSION 3.12)
project(proyecto_final_conciencia)

set(CMAKE_CXX_STANDARD 20)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

include_directories(./include)
file(GLOB source_files src/*.cpp)

add_executable(${PROJECT_NAME} ${source_files})
