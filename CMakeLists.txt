cmake_minimum_required(VERSION 3.12)
project(popn_ai)

set(CMAKE_CXX_STANDARD 20)
if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

if(MSYS)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE BOOL "" FORCE)
    set(CMAKE_DEPFILE_FLAGS_C "-MD -MT <OBJECT> -MF <DEPFILE>")
    set(CMAKE_DEPFILE_FLAGS_CXX "-MD -MT <OBJECT> -MF <DEPFILE>")

    set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)

    file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CMAKE_CURRENT_SOURCE_DIR)
endif()

add_subdirectory(generator)
add_subdirectory(tests)

include_directories(./include)
file(GLOB source_files src/*.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${source_files})

if(MSYS)
  set(NORMALIZED_OMP_LIBS "")
  foreach(_raw_lib IN LISTS OpenMP_CXX_LIBRARIES)
    string(REGEX MATCH "^([A-Za-z]):/(.*)$" _match "${_raw_lib}")
    if(_match)
      set(_drive "${CMAKE_MATCH_1}")
      set(_rest  "${CMAKE_MATCH_2}")
      string(TOLOWER "${_drive}" _drive_lc)
      set(_lib_norm "/${_drive_lc}/${_rest}")
    else()
      set(_lib_norm "${_raw_lib}")
    endif()

    message(STATUS " → normalized OMP lib: ${_lib_norm}")
    list(APPEND NORMALIZED_OMP_LIBS "${_lib_norm}")
  endforeach()
  set(omplib ${NORMALIZED_OMP_LIBS} CACHE INTERNAL "")
  target_link_libraries(${PROJECT_NAME} PUBLIC ${omplib})
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()



