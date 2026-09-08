#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "preprocess.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./classify <image_path> [model_path] [tensor_dump_path]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];

    std::vector<std::string> labels = {"NORMAL", "PNEUMONIA"};

    std::cout << "Loading image: " << image_path << std::endl;

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: could not load image at " << image_path << std::endl;
        return 1;
    }

    img = pillow_grayscale_resize(img, 224, 224);
    const float means[] = {0.485f, 0.456f, 0.406f};
    const float stds[] = {0.229f, 0.224f, 0.225f};
    std::vector<float> input_tensor_values(3 * 224 * 224);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 224; ++y)
            for (int x = 0; x < 224; ++x)
                input_tensor_values[c * 224 * 224 + y * 224 + x] =
                    (img.at<unsigned char>(y, x) / 255.0f - means[c]) / stds[c];
    if (argc > 3) {
        std::ofstream dump(argv[3], std::ios::binary);
        dump.write(reinterpret_cast<const char*>(input_tensor_values.data()),
                   input_tensor_values.size() * sizeof(float));
        if (!dump) { std::cerr << "Cannot write tensor dump\n"; return 1; }
    }
    std::cout << std::setprecision(9);
    std::cout << "Preprocessing done. Tensor size: " << input_tensor_values.size() << std::endl;

    std::cout << "First 5 C++ preprocessed values: ";
    for (int i = 0; i < 5; ++i) std::cout << input_tensor_values[i] << " ";
    double sum = 0.0, sum_sq = 0.0;
    for (float v : input_tensor_values) { sum += v; sum_sq += double(v) * v; }
    const double mean = sum / input_tensor_values.size();
    std::cout << "\nC++ tensor mean: " << mean << ", stddev: "
              << std::sqrt(std::max(0.0, sum_sq / input_tensor_values.size() - mean * mean)) << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xray_classifier");
    Ort::SessionOptions session_options;
    Ort::Session session(env, argc > 2 ? argv[2] : XRAY_MODEL_PATH, session_options);

    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    std::cout << "Inference complete." << std::endl;

    float* output = output_tensors.front().GetTensorMutableData<float>();

    int best_idx = 0;
    float best_score = output[0];
    for (size_t i = 1; i < labels.size(); i++) {
        if (output[i] > best_score) {
            best_score = output[i];
            best_idx = i;
        }
    }

    std::cout << "Raw scores -> NORMAL: " << output[0] 
               << ", PNEUMONIA: " << output[1] << std::endl;
    std::cout << "Prediction: " << labels[best_idx] 
               << " (score: " << best_score << ")" << std::endl;

    return 0;
}