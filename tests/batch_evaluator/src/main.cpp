#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <map>
#include <vector>
#include <filesystem>

#include "utec/nn/nn_optimizer.h"
#include "training/trainer.h"

using namespace training;
using namespace utec::neural_network;

std::tuple<std::vector<std::vector<float>>, std::vector<int>> load_dataset(
    const std::string& image_file,
    const std::string& label_file
) {
    std::ifstream img_f(image_file, std::ios::binary);
    uint32_t width, height, count;
    img_f.read(reinterpret_cast<char*>(&width), sizeof(width));
    img_f.read(reinterpret_cast<char*>(&height), sizeof(height));
    img_f.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::vector<std::vector<float>> images(count, std::vector<float>(width * height));
    for (uint32_t i = 0; i < count; ++i) {
        std::vector<unsigned char> buffer(width * height);
        img_f.read(reinterpret_cast<char*>(buffer.data()), width * height);
        for (size_t j = 0; j < buffer.size(); ++j)
            images[i][j] = buffer[j] / 255.0f;
    }

    std::ifstream label_f(label_file, std::ios::binary);
    uint32_t label_count;
    label_f.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
    std::vector<int32_t> labels(label_count);
    label_f.read(reinterpret_cast<char*>(labels.data()), label_count * sizeof(int32_t));

    std::cout << "Loading data is done!" << std::endl;

    return {images, labels};
}  

int main() {
    auto [images, labels] = load_dataset("../../results/test_images.bin", "../../results/test_labels.bin");

    std::map<int32_t, int> patternClass;
    std::vector<int> classPattern;
    int curClass = 0;

    for (int32_t label : labels) {
        if (patternClass.count(label) == 0) {
            patternClass[label] = curClass++;
            classPattern.push_back(label);
        }
    }

    Tensor<float, 2> input_tensor(images.size(), images[0].size());
    Tensor<float, 2> label_tensor(images.size(), classPattern.size());
    label_tensor.fill(0.0f);

    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t j = 0; j < images[i].size(); ++j) input_tensor(i, j) = images[i][j];
    }

    for (size_t i = 0; i < labels.size(); ++i) {
        int class_idx = patternClass[labels[i]];
        label_tensor(i, class_idx) = 1.0f;
    }

    Trainer net(true);

    std::cout << "Predicting..." << std::endl;

    std::vector<size_t> which_failed;
    size_t correct = 0;
    size_t incorrect = 0;

    size_t correctPerc = net.predict_batch(input_tensor, label_tensor, labels, patternClass, classPattern, which_failed, correct, incorrect, true);

    std::cout << "Final accuracy: " << correctPerc << "%" << std::endl;
    std::cout << "Correct guesses: " << correct << " | Incorrect guesses: " << incorrect << std::endl;
    std::cout << "Which failed: [";
    for (size_t &i : which_failed) {
        std::cout << i << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}