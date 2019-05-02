cmake_minimum_required(VERSION 3.14)
project(lenet)

set(CMAKE_CXX_STANDARD 17)
#include_directories("/usr/local/caffe/include")
#link_directories("/usr/include/caffe/lib")

include_directories(cmake-build-debug/include)
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")

find_package(Boost 1.70.0 REQUIRED COMPONENTS system filesystem)
include_directories(lenet ${Boost_INCLUDE_DIRS})
link_directories(lenet ${Boost_LIBRARY_DIRS})
add_executable(lenet main.cpp cmake-build-debug/include/data.h cmake-build-debug/include/data.cpp cmake-build-debug/include/maths.cpp cmake-build-debug/include/maths.h cmake-build-debug/include/lenet.cpp cmake-build-debug/include/lenet.h)

find_package(OpenCV REQUIRED)
#target_link_libraries(lenet "/usr/include/caffe/lib/libcaffe.so" )
target_link_libraries(lenet ${OpenCV_LIBS})
target_link_libraries(lenet ${Boost_LIBRARIES})

