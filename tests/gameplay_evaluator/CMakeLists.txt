cmake_minimum_required(VERSION 3.10.0)
project(gameplay_evaluator VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)


pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice)
if(LINUX)
    find_package(X11)
    if (NOT X11_FOUND)
        pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)
    endif()
endif()

include_directories(../../include)
file(GLOB source_files src/*.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${source_files})

if (LINUX)
    if (X11_FOUND)
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_X11)
        target_include_directories(${PROJECT_NAME} PRIVATE ${X11_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} X11::X11 ${X11_XTest_LIB})
        message(STATUS "X11 will be used as key backend")
    endif()
endif()

if (WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_WIN32)
endif()

target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
)

if (LINUX)
    target_link_libraries(${PROJECT_NAME} PkgConfig::PIPEWIRE)
endif()