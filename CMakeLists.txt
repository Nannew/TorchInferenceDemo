cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(TARGET_NAME torchapp)
project(${TARGET_NAME})


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    enable_language("CUDA")
    message(STATUS "Modern CMake")
else()
    find_package(CUDA 11.8 REQUIRED)
    message(STATUS "Legacy CMake - CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
	
    # Pass options to NVCC
    #https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
	set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75" )
endif()

# https://pytorch.org/cppdocs/installing.html
# libtorch-win-shared-with-deps-2.4.1+cu118
SET(Torch_DIR "O:/libtorch/share/cmake/Torch" CACHE PATH "LibTorch directory")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Use itk to load NIfTI data
SET(ITK_DIR "O:/itk/InsightToolkit-5.3.0_x64_vs14" CACHE PATH "ITK directory")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Include files
FILE( GLOB SOURCE_FILES *.cpp *.h) 

add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${TARGET_NAME} ${ITK_LIBRARIES})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
    # https://stackoverflow.com/questions/67966258/cuda-architectures-is-empty-for-target-cmtc-28d80
    # Find you GPU "Compute Capability": https://developer.nvidia.com/cuda-gpus
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "50;60;61;70;75")
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${TARGET_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${TARGET_NAME}>)
endif (MSVC)


# Organize target into a group in Visual Studio IDE
if (DEFINED TARGET_FOLDER)
	set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER ${TARGET_FOLDER})
endif()