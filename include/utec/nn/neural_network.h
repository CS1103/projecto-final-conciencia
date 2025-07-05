//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_dense.h"
#include "nn_optimizer.h"

#include <vector>
#include <memory>
#include <fstream>

namespace utec::neural_network {
    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.push_back(std::move(layer));
        };

        Tensor<T, 2> predict(const Tensor<T, 2>& X) {
            Tensor<T, 2> out = X;
            for (auto& layer : layers) out = layer->forward(out);
            return out;
        };

        void save(const std::string& path) const {
            std::ofstream out(path, std::ios::binary);
            uint32_t num_layers = layers.size();
            out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
            for (auto& layer : layers) {
                if (auto* d = dynamic_cast<Dense<T>*>(layer.get())) {
                    char type = 'D'; out.write(&type, 1);
                    auto ws = d->weight().shape(); uint32_t r = ws[0], c = ws[1];
                    out.write(reinterpret_cast<const char*>(&r), sizeof(r));
                    out.write(reinterpret_cast<const char*>(&c), sizeof(c));
                    // write weight raw data
                    out.write(reinterpret_cast<const char*>(&*d->weight().cbegin()), sizeof(T)*r*c);
                    auto bs = d->bias().shape(); uint32_t br = bs[0], bc = bs[1];
                    out.write(reinterpret_cast<const char*>(&br), sizeof(br));
                    out.write(reinterpret_cast<const char*>(&bc), sizeof(bc));
                    out.write(reinterpret_cast<const char*>(&*d->weight().cbegin()), sizeof(T)*br*bc);
                } else {
                    char type = 'O'; out.write(&type,1);
                }
            }
        }

        void load(const std::string& path) {
            std::ifstream in(path, std::ios::binary);
            uint32_t num_layers; in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
            for (uint32_t i = 0; i < num_layers; ++i) {
                char type; in.read(&type,1);
                if (type=='D') {
                    uint32_t r,c; in.read(reinterpret_cast<char*>(&r),sizeof(r)); in.read(reinterpret_cast<char*>(&c),sizeof(c));
                    Tensor<T,2> W(r,c);
                    in.read(reinterpret_cast<char*>(&*W.begin()), sizeof(T)*r*c);
                    uint32_t br,bc; in.read(reinterpret_cast<char*>(&br),sizeof(br)); in.read(reinterpret_cast<char*>(&bc),sizeof(bc));
                    Tensor<T,2> B(br,bc);
                    in.read(reinterpret_cast<char*>(&*B.begin()), sizeof(T)*br*bc);
                    auto* d = dynamic_cast<Dense<T>*>(layers[i].get());
                    d->weight() = W; d->bias() = B;
                }
            }
        }

        template <template <typename ...> class LossType, template <typename ...> class OptimizerType = SGD>
        void train(
            const Tensor<T,2>& X,
            const Tensor<T,2>& Y,
            const size_t epochs,
            const size_t batch_size,
            T learning_rate
        ) {
            OptimizerType<T> optimizer(learning_rate);
            const size_t N = X.shape()[0];
            const size_t C = X.shape()[1];

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                std::cout << "[Epoch " << (epoch+1) << "/" << epochs << "] starting with learning rate " << optimizer.learning_rate() << "...\n";

                for (size_t start = 0; start < N; start += batch_size) {
                    size_t end = std::min(start + batch_size, N);
                    size_t B = end - start;

                    Tensor<T,2> Xb(B, C), Yb(B, Y.shape()[1]);
                    for (size_t i = 0; i < B; ++i) {
                        for (size_t j = 0; j < C; ++j) {
                            Xb(i,j) = X(start + i, j);
                        }
                        for (size_t j = 0; j < Y.shape()[1]; ++j) {
                            Yb(i,j) = Y(start + i, j);
                        }
                    }

                    Tensor<T,2> preds = predict(Xb);
                    LossType<T> batch_loss(preds, Yb);
                    Tensor<T,2> grad = batch_loss.loss_gradient();
                    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                        grad = (*it)->backward(grad);
                    for (auto& layer : layers)
                        layer->update_params(optimizer);
                }

                Tensor<T,2> ep_preds = predict(X);
                LossType<T> epoch_loss(ep_preds, Y);
                T    loss_val = epoch_loss.loss();

                size_t correct = 0;
                size_t M = ep_preds.shape()[0];
                size_t K = ep_preds.shape()[1];
                for (size_t i = 0; i < M; ++i) {
                    size_t best_j = 0;
                    T best_v = ep_preds(i,0);
                    for (size_t j = 1; j < K; ++j) {
                        if (ep_preds(i,j) > best_v) {
                            best_v = ep_preds(i,j);
                            best_j = j;
                        }
                    }
                    if (Y(i,best_j) == T(1)) ++correct;
                }
                float acc = 100.0f * correct / static_cast<float>(M);

                std::cout << "  Loss: " << loss_val
                        << "% | Accuracy: " << acc << "%\n\n";
            }
        };
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H