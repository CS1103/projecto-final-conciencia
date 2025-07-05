cmake_minimum_required(VERSION 3.10.0)
project(gameplay_evaluator VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)
pkg_check_modules(GLIB REQUIRED glib-2.0 gio-2.0)
pkg_check_modules(UUID REQUIRED uuid)

link_directories(${GLIB_LIBRARY_DIRS})

include_directories(../../include)
file(GLOB source_files src/*.cpp)

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${PIPEWIRE_INCLUDE_DIRS}
	${SPA_INCLUDE_DIRS} 
    ${GLIB_INCLUDE_DIRS}
    ${UUID_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} ${source_files})
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS}
    PkgConfig::PIPEWIRE
    ${GLIB_LIBRARIES}
    ${UUID_LIBRARIES}
)   