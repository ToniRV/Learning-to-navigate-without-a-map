cmake_minimum_required(VERSION 2.6)

# try to force using gcc
# SET(CMAKE_C_COMPILER gcc)
# SET(CMAKE_CXX_COMPILER g++)

project(DSTAR_LITE)
# Make sure we build with c++11 support.
add_definitions(-std=c++11)

# Make sure warnings are reported and treated as errors.
# UNCOMMENT THIS, the dstar is full of warnings...
#add_definitions(-Wall -Werror)

SET(OS_UNIX FALSE)
SET(OS_LINUX FALSE)
SET(OS_MACOSX FALSE)

IF (UNIX)
    SET(OS_UNIX TRUE)
ENDIF()

IF (UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
	SET(OS_LINUX TRUE)
    MESSAGE(STATUS "This is Linux")
    add_definitions(-DUSE_OPEN_GL=TRUE)
    find_package(GLUT)
    find_package(OpenGL)
ELSEIF (UNIX AND APPLE AND "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
	SET(OS_MACOSX TRUE)
    add_definitions(-DMACOS=1)
    ADD_DEFINITIONS(-DUSE_OPEN_GL=0)
    MESSAGE(STATUS "This is Mac")
    # do not use opengl in mac
ENDIF()

set(CMAKE_CXX_FLAGS "-O2 -g -Wall")

include_directories(include)
link_directories(${DSTAR_LITE_BINARY_DIR})

if (UNIX AND APPLE AND "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    message(STATUS "Build without OpenGL")
    add_library(dstar_lite src/dstar.cpp)
    target_link_libraries(dstar_lite)
elseif(UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    message(STATUS "Build with OpenGL")
    add_library(dstar_lite src/dstar.cpp)
    target_link_libraries(dstar_lite ${GLUT_LIBRARY} ${OPENGL_LIBRARY})
    add_executable(dstar_draw src/dstar.cpp src/dstar_draw.cpp)
    target_link_libraries(dstar_draw dstar_lite)
    add_executable(dstar_from_grid src/dstar_from_grid.cpp)
    target_link_libraries(dstar_from_grid dstar_lite)
endif()

add_executable(dstar_from_input src/dstar_from_input.cpp)
target_link_libraries(dstar_from_input dstar_lite)
