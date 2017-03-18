cmake_minimum_required(VERSION 2.6)
project(DSTAR_LITE)

# Make sure we build with c++11 support.
add_definitions(-std=c++11)

# Make sure warnings are reported and treated as errors.
# UNCOMMENT THIS, the dstar is full of warnings...
#add_definitions(-Wall -Werror)

add_definitions(-DUSE_OPEN_GL)

find_package(GLUT)
find_package(OpenGL)

set(CMAKE_CXX_FLAGS "-O2 -g -Wall")

include_directories(include)
link_directories(${DSTAR_LITE_BINARY_DIR})

add_library(dstar_lite src/dstar.cpp)
target_link_libraries(dstar_lite ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

add_executable(dstar_draw src/dstar.cpp src/dstar_draw.cpp)
target_link_libraries(dstar_draw dstar_lite)

add_executable(dstar_from_grid src/dstar_from_grid.cpp)
target_link_libraries(dstar_from_grid dstar_lite)
