#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

// Separable antialiased bilinear filtering, with Pillow's 22-bit weights
// and uint8 rounding after each pass. cv::INTER_LINEAR does not antialias.
struct ResizeWeights {
    int start;
    std::vector<int> weights;
};
inline std::vector<ResizeWeights> resize_weights(int source, int target) {
    std::vector<ResizeWeights> result;
    const double scale = double(source) / target;
    const double support = std::max(1.0, scale);
    for (int i = 0; i < target; ++i) {
        const double center = (i + 0.5) * scale;
        const int start = std::max(0, int(center - support + 0.5));
        const int end = std::min(source, int(center + support + 0.5));
        std::vector<double> weights;
        double total = 0;
        for (int j = start; j < end; ++j) {
            const double weight = std::max(0.0, 1.0 - std::abs((j - center + 0.5) / support));
            weights.push_back(weight);
            total += weight;
        }
        ResizeWeights entry{start, {}};
        for (double w : weights) entry.weights.push_back(int(w / total * (1 << 22) + 0.5));
        result.push_back(entry);
    }
    return result;
}
inline cv::Mat pillow_grayscale_resize(const cv::Mat& bgr, int width, int height) {
    cv::Mat gray(bgr.rows, bgr.cols, CV_8U);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            const auto p = bgr.at<cv::Vec3b>(y, x);
            gray.at<unsigned char>(y, x) = (19595 * p[2] + 38470 * p[1] + 7471 * p[0] + 32768) >> 16;
        }
    const auto horizontal = resize_weights(gray.cols, width);
    const auto vertical = resize_weights(gray.rows, height);
    cv::Mat temp(gray.rows, width, CV_8U), output(height, width, CV_8U);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < width; ++x) {
            const auto& w = horizontal[x];
            int sum = 1 << 21;
            for (size_t k = 0; k < w.weights.size(); ++k)
                sum += gray.at<unsigned char>(y, w.start + k) * w.weights[k];
            temp.at<unsigned char>(y, x) = std::clamp(sum >> 22, 0, 255);
        }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const auto& w = vertical[y];
            int sum = 1 << 21;
            for (size_t k = 0; k < w.weights.size(); ++k)
                sum += temp.at<unsigned char>(w.start + k, x) * w.weights[k];
            output.at<unsigned char>(y, x) = std::clamp(sum >> 22, 0, 255);
        }
    return output;
}
