#include <iostream>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <string>

#include "training/trainer.h"

#include "utils/number.h"
#include "utils/print.h"

#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace utec::neural_network;
using namespace popn::utils;
using namespace training;

std::vector<int32_t> combinations;
std::random_device rd;
std::mt19937 gen(rd());

int main() {
    Tensor<float, 2> imageData(1, 154 * 13);
    cv::Mat image;

    for (int r = 1; r <= 9; ++r) {
        std::vector<int> current;
        generate_combos(r, 0, current, combinations);
    }

    combinations.push_back(0);

    Trainer net(true);

    std::string custom_path = "";
    bool running = true;

    do {
        std::cout << "Input a directory (Send R key to use default path): " << std::endl;
        std::getline(std::cin, custom_path);
        bool useDefault = custom_path.size() == 1 && std::tolower(custom_path[0]) == 'r';
        if (!useDefault) {
            if (!std::filesystem::is_directory(custom_path) || !std::filesystem::exists(custom_path)) {
                std::cout << "Invalid directory. Try again." << std::endl;
                custom_path = "";
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

        std::cout << "Type filename + extension to predict (Send E key to exit): " << std::endl;
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
            
            size_t pred = net.predict_image(imageData);
            print_ascii(image, 154*13);
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