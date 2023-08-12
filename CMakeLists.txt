cmake_minimum_required(VERSION 3.19)
project(learning_triple_poses)

set(CMAKE_CXX_STANDARD 14)
#set( CMAKE_BUILD_TYPE Release )
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(inc)

add_executable(learning_triple_poses main.cpp src/learning_triple_poses.cpp src/homotopy.cpp)

add_executable(sift example/sift_extraction.cpp)
target_link_libraries(sift ${OpenCV_LIBS})
