#include <iostream>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <string>

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_dense.h"

#include <random>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

using namespace utec::neural_network;

std::vector<int> elements = {1,2,3,4,5,6,7,8,9};
std::vector<int32_t> combinations;
std::random_device rd;
std::mt19937 gen(rd());

void generate_combos(const std::vector<int> elms, int r, int st, std::vector<int> &cur, std::vector<int32_t> &res) {
    if (cur.size() == r) {
        std::string number_str;
        for (int num : cur) number_str += std::to_string(num);
        res.push_back(std::stoi(number_str));
        return;
    }

    for (int i = st; i < elements.size(); ++i) {
        cur.push_back(elements[i]);
        generate_combos(elements, r, i + 1, cur, res);
        cur.pop_back();
    }    
}

std::string gscale = " .,:oOX#$@";

int main() {
    Tensor<float, 2> imageData(1, 154 * 13);
    cv::Mat image;

    for (int r = 1; r <= 9; ++r) {
        std::vector<int> current;
        generate_combos(elements, r, 0, current, combinations);
    }

    combinations.push_back(0);

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
    net.load("../../model_ep250.nn");
    std::cout << "Done." << std::endl;

    std::string custom_path = "";
    bool running = true;

    do {
        std::cout << "Input a directory (Send R key to use default path): " << std::endl;
        std::getline(std::cin, custom_path);
        bool useDefault = custom_path.size() == 1 && std::tolower(custom_path[0]) == 'r';
        if (!useDefault) {
            if (!std::filesystem::is_directory(custom_path) || !std::filesystem::exists(custom_path)) {
                std::cout << "Invalid directory. Try again." << std::endl;
            }
        }
        else {
            std::cout << "Reading from results directory (../../results/images)..." << std::endl;
            custom_path = "../../results/images";
            if (!std::filesystem::is_directory("../../results/images") || !std::filesystem::exists("../../results/images")) {
                throw std::runtime_error("Results directory not found.");
            }
        }
    } while (custom_path.empty());
    
    do {
        std::string filename, fullpath;

        std::cout << "Type filename + extension to predict (or send E key to exit): " << std::endl;
        std::getline(std::cin, filename);
        fullpath = custom_path + "/" + filename;
        if (!std::filesystem::is_directory(fullpath) && std::filesystem::exists(fullpath)) {
            try {
                image = cv::imread(fullpath, cv::IMREAD_GRAYSCALE);
                for (size_t i = 0; i < 154; ++i) {
                    for (size_t j = 0; j < 13; ++j) imageData(0, i*j) = image.data[i*j];
                }
            } catch (...) {
                std::cout << "File could not be opened/parsed. Perhaps you're using the wrong format?" << std::endl;
            }
            
            auto final_pred = net.predict(imageData);
            size_t pred = 0;
            auto maxv = final_pred(0,0);
            
            for (size_t j = 1; j < final_pred.shape()[1]; ++j) {
                if (final_pred(0,j) > maxv) { maxv = final_pred(0,j); pred = j; }
            }
            
            for (size_t i = 0; i < 154*13; i++) {
                int char_index = static_cast<int>((static_cast<double>(image.data[i]) / 255) * (gscale.length() - 1));
                std::cout << gscale[char_index];
                if ((i + 1) % 154 == 0) std::cout << std::endl;
            }

            std::cout << " Predicted pattern: " << combinations[pred] << std::endl << std::endl;
        }
        else {
            if (filename.size() == 1 && std::tolower(filename[0]) == 'e') {
                std::cout << "Exiting..."  << std::endl;
                running = false;
            } else std::cout << "Invalid filename/path. Try again." << std::endl;
        }
    } while (running);

    return 0;
}