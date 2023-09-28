cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(AnACor LANGUAGES CXX)  # Changed project name
set(AnACor_VERSION_ID "0x000001")  # Changed variable prefix
set(AnACor_VERSION_MAJOR 0)  # Changed variable prefix
set(AnACor_VERSION_MINOR 1)  # Changed variable prefix
set(AnACor_VERSION_PATCH 0)  # Changed variable prefix
set(AnACor_VERSION_SUFFIX "")  # Changed variable prefix

set(CMAKE_CXX_STANDARD 11)

# Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

# Get string length of current source directory for log messages.
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

# Set install directory paths.
if (NOT AnACor_LIB_INSTALL_DIR)  # Changed variable prefix
    set(AnACor_LIB_INSTALL_DIR "lib")  # Changed variable prefix
endif()
if (NOT AnACor_INCLUDE_INSTALL_DIR)  # Changed variable prefix
    set(AnACor_INCLUDE_INSTALL_DIR "include")  # Changed variable prefix
endif()
if (NOT AnACor_BIN_INSTALL_DIR)  # Changed variable prefix
    set(AnACor_BIN_INSTALL_DIR "bin")  # Changed variable prefix
endif()

find_package(CUDA 11.0 REQUIRED)
find_package(OpenMP QUIET)

# Set compiler options.
include(compiler_options)
enable_testing()

# Generate source list from subdirectories.
add_subdirectory(src)

# Build library using list of source files.
if (CUDA_FOUND)
    # Add CUDA kernel sources.
    list(APPEND ${CMAKE_PROJECT_NAME}_SRC ${${CMAKE_PROJECT_NAME}_CUDA_SRC})
    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${CMAKE_PROJECT_NAME}.dir/src)
    cuda_add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_CUFFT_LIBRARIES})
    add_definitions( -DCUDA_FOUND )
else()
    #add_library(${CMAKE_PROJECT_NAME} ${${CMAKE_PROJECT_NAME}_SRC})
    message("CUDA not found! Cannot compile library.")
endif()
if (NOT DEFINED PYTHON_EXECUTABLE)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        SOVERSION ${AnACor_VERSION_ID}  # Changed variable prefix
        VERSION ${AnACor_VERSION_ID})  # Changed variable prefix
else()
    message(STATUS "Building for Python: ${PYTHON_EXECUTABLE}")
endif()

# # Build example application.
# add_subdirectory(application)

# # Build tests.
# add_subdirectory(tests)

# target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_CUFFT_LIBRARIES})
# target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX)

# install(TARGETS ${CMAKE_PROJECT_NAME}
#     ARCHIVE DESTINATION ${AnACor_LIB_INSTALL_DIR} COMPONENT libraries  # Changed variable prefix
#     LIBRARY DESTINATION ${AnACor_LIB_INSTALL_DIR} COMPONENT libraries  # Changed variable prefix
#     RUNTIME DESTINATION ${AnACor_BIN_INSTALL_DIR} COMPONENT runtime  # Changed variable prefix
# )
# install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/include"
#     DESTINATION ${AnACor_INCLUDE_INSTALL_DIR}  # Changed variable prefix
#     COMPONENT headers
#     FILES_MATCHING REGEX "(.*h)|(.*hpp)"
# )
