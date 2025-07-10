#include <iostream>
#include <random>
#include <filesystem>
#include <map>
#include <vector>
#include <chrono>

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_dense.h"

using namespace utec::neural_network;

namespace training
{
    static auto init_w = [](auto &W)
    {
        std::mt19937 gen(42);
        float fan_in = W.shape()[1];
        float fan_out = W.shape()[0];
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto &v : W)
            v = dist(gen);
    };

    static auto init_b = [](auto &B)
    {
        for (auto &val : B)
            val = 0.0f;
    };

    class Trainer {
        NeuralNetwork<float> net; // trabajamos con floats
        std::string customPath;
    public:
        Trainer(bool isTrained = false)
        {
            net.add_layer(std::make_unique<Dense<float>>(154 * 13, 128, init_w, init_b));
            net.add_layer(std::make_unique<ReLU<float>>());
            net.add_layer(std::make_unique<Dense<float>>(128, 64, init_w, init_b));
            net.add_layer(std::make_unique<ReLU<float>>());
            net.add_layer(std::make_unique<Dense<float>>(64, 512, init_w, init_b));
            net.add_layer(std::make_unique<Softmax<float>>());

            if (isTrained)
            {
                std::string ai_path = "";
                do
                {
                    std::cout << "Input full path of model: " << std::endl;
                    std::getline(std::cin, ai_path);
                    if (std::filesystem::is_directory(ai_path) || !std::filesystem::exists(ai_path))
                    {
                        std::cout << "Invalid path. Try again." << std::endl;
                        ai_path = "";
                    }
                } while (ai_path.empty());

                customPath = ai_path;

                std::cout << "Reading model from file... ";
                net.load(ai_path);
                std::cout << "Done." << std::endl;
            }
        }

        [[nodiscard]] std::string custom_path() const { return customPath; };

        void save_model(std::string location){
            std::cout << "Saving to file... ";
            net.save(location);
            std::cout << "Done." << std::endl;
        }

        void callTrain(Tensor<float, 2> input, Tensor<float, 2> label, int epochs) {
            auto t0 = std::chrono::steady_clock::now();
            net.train<CrossEntropyLoss>(input, label, epochs, 32, 0.001f);
            auto t1 = std::chrono::steady_clock::now();
            std::cout << "Training took "
                    << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()
                    << " seconds\n";
        }

        [[nodiscard("This is a prediction. Do not discard.")]] size_t predict_image(Tensor<float, 2> imageData) {
            auto final_pred = net.predict(imageData);
            size_t pred = 0;
            auto maxv = final_pred(0, 0);

            for (size_t j = 1; j < final_pred.shape()[1]; ++j)
            {
                if (final_pred(0, j) > maxv)
                {
                    maxv = final_pred(0, j);
                    pred = j;
                }
            }
            return pred;
        };

        [[nodiscard("This is a prediction. Do not discard.")]] size_t predict_test(Tensor<float, 2> inputData, bool perc = false) {
            auto final_preds = net.predict(inputData);
            size_t correct = 0;
            for (size_t i = 0; i < final_preds.shape()[0]; ++i)
            {
                size_t pred = 0;
                auto maxv = final_preds(i, 0);
                for (size_t j = 1; j < final_preds.shape()[1]; ++j)
                {
                    if (final_preds(i, j) > maxv)
                    {
                        maxv = final_preds(i, j);
                        pred = j;
                    }
                }
                if (inputData(i, pred) == 1.0f)
                    ++correct;
            }
            if (!perc) return correct;
            else return (100.0f * correct / final_preds.shape()[0]);
        };

        [[nodiscard("This is a prediction. Do not discard.")]] size_t predict_batch(Tensor<float, 2> inputData, Tensor<float, 2> labelData, std::vector<int> &labels, std::map<int32_t, int> &patternClass, std::vector<int> &classPattern, std::vector<size_t> &whichfail, size_t &correct, size_t &incorrect, bool perc = false) {
            auto final_preds = net.predict(inputData);
            for (size_t i = 0; i < final_preds.shape()[0]; ++i) {
                size_t pred = 0;
                auto maxv = final_preds(i,0);
                for (size_t j = 1; j < final_preds.shape()[1]; ++j) {
                    if (final_preds(i,j) > maxv) { maxv = final_preds(i,j); pred = j; }
                }
                if (labelData(i, pred) == 1.0f) correct++;
                else {
                    incorrect++;
                    whichfail.push_back(i);
                }

                std::cout << "[" << i << "] ";
                std::cout << "Real pattern: " << patternClass[labels[i]] << " (" << labels[i] << ")" << std::endl;
                std::cout << " Predicted pattern: " << patternClass[classPattern[pred]] << " (" << classPattern[pred] << ")" << std::endl << std::endl;
            }
            if (!perc) return correct;
            else return (100.0f * correct / final_preds.shape()[0]);
        };
    };
};