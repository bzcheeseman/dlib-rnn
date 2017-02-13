cmake_minimum_required(VERSION 3.7)
project(dlib_rnn)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(USE_CUDA ON)
set(COMPILER_CAN_DO_CPP_11 1)
add_definitions(-DDLIB_USE_CUDA)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=8 -march=native -mtune=native -O3")

include_directories("/usr/local/include/eigen3")

#include(cmake_utils/add_global_compiler_switch.cmake)
#
#macro (toggle_preprocessor_switch option_name)
#    if (${option_name})
#        add_global_define(${option_name})
#    else()
#        remove_global_define(${option_name})
#    endif()
#endmacro()
#
#if (USE_CUDA)
    find_package(CUDA 7.5)
#
#    if (CUDA_FOUND AND MSVC AND NOT CUDA_CUBLAS_LIBRARIES AND "${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
#        message(WARNING "You have CUDA installed, but we can't use it unless you put visual studio in 64bit mode.")
#        set(CUDA_FOUND 0)
#    endif()
#
#    if (CUDA_FOUND AND COMPILER_CAN_DO_CPP_11)
#
#        set(CUDA_HOST_COMPILATION_CPP ON)
#        # Note that we add __STRICT_ANSI__ to avoid freaking out nvcc with gcc specific
#        # magic in the standard C++ header files (since nvcc uses gcc headers on
#        # linux).
#        list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")
#
#        include(cmake_utils/test_for_cudnn/find_cudnn.txt)
#
#    endif()
#
#    if (CUDA_FOUND AND cudnn AND cudnn_include AND COMPILER_CAN_DO_CPP_11)
#        set(SOURCE_FILES ${SOURCE_FILES}
#                dlib/dnn/cuda_dlib.cu
#                dlib/dnn/cudnn_dlibapi.cpp
#                dlib/dnn/cublas_dlibapi.cpp
#                dlib/dnn/curand_dlibapi.cpp
#                dlib/dnn/gpu_data.cpp
#                )
#        set(dlib_needed_libraries ${dlib_needed_libraries}
#                ${CUDA_CUBLAS_LIBRARIES}
#                ${cudnn}
#                ${CUDA_curand_LIBRARY}
#                )
#        include_directories(${cudnn_include})
#    else()
#        set(USE_CUDA OFF)
#        toggle_preprocessor_switch(DLIB_USE_CUDA)
#        if (COMPILER_CAN_DO_CPP_11)
#            if (NOT cudnn OR NOT cudnn_include OR NOT cudnn_test_compile_worked)
#                message(STATUS "*** cuDNN V5.0 OR GREATER NOT FOUND.  DLIB WILL NOT USE CUDA. ***")
#                message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.")
#            endif()
#        else ()
#            message(STATUS "*** Dlib CUDA support requires C++11 but your compiler doesn't support it. ***")
#        endif()
#    endif()
#endif()

set(SOURCE_FILES ${SOURCE_FILES}
        include/cudnn_rnn.hpp
#        include/rnn.h
#        include/input_one_hot.h
#        main.cpp
        include/binary.hpp
        main2.cpp include/ntm.hpp)

cuda_add_executable(dlib_rnn ${SOURCE_FILES})

#add_executable(dlib_rnn ${SOURCE_FILES})
target_link_libraries(dlib_rnn /usr/local/cuda/lib/libcudnn.dylib /usr/local/cuda/lib/libcurand.dylib dlib)