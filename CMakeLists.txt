cmake_minimum_required(VERSION 3.15)
project(XrayClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(/usr/local/opt/onnxruntime/include/onnxruntime)
link_directories(/usr/local/opt/onnxruntime/lib)

add_executable(classify main.cpp)

target_include_directories(classify PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(classify PRIVATE ${OpenCV_LIBS} onnxruntime)
target_compile_definitions(classify PRIVATE XRAY_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_v3_xray.onnx")
