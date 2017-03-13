cmake_minimum_required(VERSION 2.6)
project(DSTAR_LITE)

add_definitions(-DUSE_OPEN_GL)

find_package(GLUT)
find_package(OpenGL)

set(CMAKE_CXX_FLAGS "-O2 -g -Wall")

include_directories(${DSTAR_LITE_SOURCE_DIR})
link_directories(${DSTAR_LITE_BINARY_DIR})

add_library(dstar_lite dstar.cpp)
target_link_libraries(dstar_lite ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

add_executable(dstar_draw dstar.cpp dstar_draw.cpp)
target_link_libraries(dstar_draw dstar_lite) #link the math library
