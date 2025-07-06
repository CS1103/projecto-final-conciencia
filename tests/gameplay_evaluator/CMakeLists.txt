cmake_minimum_required(VERSION 3.10.0)
project(gameplay_evaluator VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(X11)

pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice)
pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)

include_directories(../../include)
file(GLOB source_files src/*.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${source_files})

if (X11_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_X11)
    target_include_directories(${PROJECT_NAME} PRIVATE ${X11_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} X11::X11 ${X11_XTest_LIB})
    message(STATUS "X11 will be used as key backend")
endif()

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    PkgConfig::PIPEWIRE
)