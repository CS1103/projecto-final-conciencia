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

std::tuple<std::vector<std::vector<float>>, std::vector<int32_t>> load_dataset(
    const std::string &image_file,
    const std::string &label_file)
{
    std::ifstream img_f(image_file, std::ios::binary);
    uint32_t width, height, count;
    img_f.read(reinterpret_cast<char *>(&width), sizeof(width));
    img_f.read(reinterpret_cast<char *>(&height), sizeof(height));
    img_f.read(reinterpret_cast<char *>(&count), sizeof(count));

    std::vector<std::vector<float>> images(count, std::vector<float>(width * height));
    for (uint32_t i = 0; i < count; ++i)
    {
        std::vector<unsigned char> buffer(width * height);
        img_f.read(reinterpret_cast<char *>(buffer.data()), width * height);
        for (size_t j = 0; j < buffer.size(); ++j)
            images[i][j] = buffer[j] / 255.0f;
    }

    std::ifstream label_f(label_file, std::ios::binary);
    uint32_t label_count;
    label_f.read(reinterpret_cast<char *>(&label_count), sizeof(label_count));
    std::vector<int32_t> labels(label_count);
    label_f.read(reinterpret_cast<char *>(labels.data()), label_count * sizeof(int32_t));

    std::cout << "Loading data is done!" << std::endl;

    return {images, labels};
}

int mainTraining(int epochs)
{
    auto [images, labels] = load_dataset("./results/images.bin", "./results/labels.bin");
    auto [test_images, test_labels] = load_dataset("./results/test_images.bin", "./results/test_labels.bin");

    std::map<int32_t, int> patternClass, patternTestClass;
    std::vector<int> classPattern;
    int curClass = 0;

    for (int32_t label : labels)
    {
        if (patternClass.count(label) == 0)
        {
            patternClass[label] = curClass++;
            classPattern.push_back(label);
        }
    }

    curClass = 0;
    for (int32_t label : test_labels)
    {
        if (patternTestClass.count(label) == 0)
            patternTestClass[label] = curClass++;
    }

    Tensor<float, 2> input_tensor(images.size(), images[0].size());
    Tensor<float, 2> label_tensor(images.size(), classPattern.size());
    label_tensor.fill(0.0f);

    Tensor<float, 2> testinp_tensor(test_images.size(), test_images[0].size());
    Tensor<float, 2> testlab_tensor(test_images.size(), classPattern.size());
    testlab_tensor.fill(0.0f);

    for (size_t i = 0; i < images.size(); ++i)
    {
        for (size_t j = 0; j < images[i].size(); ++j)
            input_tensor(i, j) = images[i][j];
    }
    for (size_t i = 0; i < test_images.size(); ++i)
    {
        for (size_t j = 0; j < test_images[i].size(); ++j)
            testinp_tensor(i, j) = test_images[i][j];
    }

    for (size_t i = 0; i < labels.size(); ++i)
    {
        int class_idx = patternClass[labels[i]];
        label_tensor(i, class_idx) = 1.0f;
    }
    for (size_t i = 0; i < test_labels.size(); ++i)
    {
        int class_idx = patternTestClass[test_labels[i]];
        testlab_tensor(i, class_idx) = 1.0f;
    }

    Trainer net;
    std::cout << "Starting to train..." << std::endl;
    net.callTrain(input_tensor, label_tensor, epochs);

    net.save_model("./model_ep" + std::to_string(epochs) + ".nn");
    std::cout << "Predicting..." << std::endl;

    size_t correctPerc = net.predict_test(testinp_tensor, true);
    std::cout << "Final accuracy: " << correctPerc << "%" << std::endl;

    return 0;
}

int redoTraining(int epochs)
{
    Trainer net(true);

    auto [images, labels] = load_dataset("./results/images.bin", "./results/labels.bin");
    auto [test_images, test_labels] = load_dataset("./results/test_images.bin", "./results/test_labels.bin");

    std::map<int32_t, int> patternClass, patternTestClass;
    std::vector<int> classPattern;
    int curClass = 0;

    for (int32_t label : labels)
    {
        if (patternClass.count(label) == 0)
        {
            patternClass[label] = curClass++;
            classPattern.push_back(label);
        }
    }

    curClass = 0;
    for (int32_t label : test_labels)
    {
        if (patternTestClass.count(label) == 0)
            patternTestClass[label] = curClass++;
    }

    Tensor<float, 2> input_tensor(images.size(), images[0].size());
    Tensor<float, 2> label_tensor(images.size(), classPattern.size());
    label_tensor.fill(0.0f);

    Tensor<float, 2> testinp_tensor(test_images.size(), test_images[0].size());
    Tensor<float, 2> testlab_tensor(test_images.size(), classPattern.size());
    testlab_tensor.fill(0.0f);

    for (size_t i = 0; i < images.size(); ++i)
    {
        for (size_t j = 0; j < images[i].size(); ++j)
            input_tensor(i, j) = images[i][j];
    }
    for (size_t i = 0; i < test_images.size(); ++i)
    {
        for (size_t j = 0; j < test_images[i].size(); ++j)
            testinp_tensor(i, j) = test_images[i][j];
    }

    for (size_t i = 0; i < labels.size(); ++i)
    {
        int class_idx = patternClass[labels[i]];
        label_tensor(i, class_idx) = 1.0f;
    }
    for (size_t i = 0; i < test_labels.size(); ++i)
    {
        int class_idx = patternTestClass[test_labels[i]];
        testlab_tensor(i, class_idx) = 1.0f;
    }

    std::cout << "Starting to train..." << std::endl;
    net.callTrain(input_tensor, label_tensor, epochs);

    size_t correctPerc = net.predict_test(testinp_tensor, true);
    std::cout << "Final accuracy: " << correctPerc << "%" << std::endl;

    std::cout << "Saving to file... ";
    net.save_model("./" + std::filesystem::path(net.custom_path()).stem().string() + "_ep" + std::to_string(epochs) + ".nn");
    std::cout << "Done." << std::endl;

    return 0;
}

int main()
{
    int opt = 0;
    std::cout << "- TRAINING MENU -" << std::endl;
    std::cout << "1) Train from zero (n epochs)" << std::endl;
    std::cout << "2) Train from trained model (n epochs)" << std::endl;
    std::cout << "3) Exit" << std::endl
              << std::endl;
    while (opt < 1 || opt > 3)
    {
        std::cin.clear();
        std::cout << "Enter a menu option: ";
        std::cin >> opt;
        if (opt > 3 || opt < 1 || std::cin.fail()) {
            std::cout << "Invalid option. Try again." << std::endl << std::endl;
            std::cin.clear();
            std::cin.ignore(80, '\n');
            opt = 0;
        }
    }

    std::cout << std::endl;

    if (opt < 3)
    {
        int eps;
        do {
            std::cin.clear();
            std::cin.ignore(80, '\n');
            std::cout << "Enter amount of epochs: ";
            std::cin >> eps;
            if (std::cin.fail() || eps < 1)
            {
                std::cout << "Please enter a valid amount." << std::endl;
            }
        } while (std::cin.fail() || eps < 1);
        
        if (opt == 1)
            mainTraining(eps);
        else
            redoTraining(eps);
        return 0;
    }
    else
        return 0;
}