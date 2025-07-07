cmake_minimum_required(VERSION 3.12)
project(popn_ai)

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

if(WIN32)
    foreach(_raw_lib IN LISTS OpenMP_CXX_LIBRARIES)
        file(TO_CMAKE_PATH "${_raw_lib}" _lib_norm)
        list(APPEND NORMALIZED_OMP_LIBS "${_lib_norm}")
        message(STATUS "${_raw_lib}")
    endforeach()
    set(omplib ${NORMALIZED_OMP_LIBS} CACHE INTERNAL "")
endif()

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} PUBLIC ${omplib})
