cmake_minimum_required(VERSION 3.16)

project("gemma_inference")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(gemma_inference SHARED gemma_inference.cpp)
target_include_directories(gemma_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ANDROID)
    set(ONNXRUNTIME_AAR_DIR "" CACHE PATH "Unpacked onnxruntime-android AAR directory")
    set(ONNXRUNTIME_HEADERS_DIR
            ${ONNXRUNTIME_AAR_DIR}/headers)
    set(ONNXRUNTIME_LIBRARY
            ${ONNXRUNTIME_AAR_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so)

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY})

    target_include_directories(gemma_inference PRIVATE ${ONNXRUNTIME_HEADERS_DIR})

    target_link_libraries(gemma_inference
            onnxruntime
            android
            log)
else()
    set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root")
    find_package(JNI REQUIRED)

    find_path(ONNXRUNTIME_INCLUDE_DIR
            onnxruntime/onnxruntime_cxx_api.h
            PATHS
            ${ONNXRUNTIME_ROOT}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            /usr/include/onnxruntime
            /usr/local/include
    )

    find_library(ONNXRUNTIME_LIBRARY
            onnxruntime
            PATHS
            ${ONNXRUNTIME_ROOT}/lib
            /usr/lib
            /usr/lib64
            /usr/local/lib
    )

    if (NOT ONNXRUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find ONNX Runtime include directory.")
    endif()

    if (NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "Cannot find ONNX Runtime library.")
    endif()

    target_include_directories(gemma_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${JNI_INCLUDE_DIRS})
    target_link_libraries(gemma_inference
            ${ONNXRUNTIME_LIBRARY}
            ${JNI_LIBRARIES})
endif()
