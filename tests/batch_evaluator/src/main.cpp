#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <map>
#include <vector>
#include <chrono>

#include "utec/algebra/Tensor.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"

#include <random>

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

    auto init_w = [&](auto& W){
        std::mt19937 gen(42);
        float fan_in  = W.shape()[1];
        float fan_out = W.shape()[0];
        float scale   = std::sqrt(2.0f/(fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& v : W) v = dist(gen);
    };

    auto init_b = [](auto& B) {
        for (auto& val : B) val = 0.0f;
    };

    NeuralNetwork<float> net;

    net.add_layer(std::make_unique<Dense<float>>(154*13, 128, init_w, init_b));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(128, 64, init_w, init_b));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(64, 512, init_w, init_b));
    net.add_layer(std::make_unique<Softmax<float>>());

    std::cout << "Reading model from file... ";
    net.load("../../model_ep100.nn");
    std::cout << "Done." << std::endl;

    std::cout << "Predicting..." << std::endl;


    std::vector<size_t> which_failed;
    auto final_preds = net.predict(input_tensor);
    size_t correct = 0;
    size_t incorrect = 0;
    for (size_t i = 0; i < final_preds.shape()[0]; ++i) {
        size_t pred = 0;
        auto maxv = final_preds(i,0);
        for (size_t j = 1; j < final_preds.shape()[1]; ++j) {
            if (final_preds(i,j) > maxv) { maxv = final_preds(i,j); pred = j; }
        }
        if (label_tensor(i, pred) == 1.0f) correct++;
        else {
            incorrect++;
            which_failed.push_back(i);
        }

        std::cout << "[" << i << "] ";
        std::cout << "Real pattern: " << patternClass[labels[i]] << " (" << labels[i] << ")" << std::endl;
        std::cout << " Predicted pattern: " << patternClass[classPattern[pred]] << " (" << classPattern[pred] << ")" << std::endl << std::endl;
    }
    std::cout << "Final accuracy: "
            << (100.0f * correct / final_preds.shape()[0])
            << "%\n";
    std::cout << "Correct guesses: " << correct << " | Incorrect guesses: " << incorrect << std::endl;
    std::cout << "Which failed: [";
    for (size_t &i : which_failed) {
        std::cout << i << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}