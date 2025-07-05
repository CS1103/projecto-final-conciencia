//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>
#include <random>

namespace utec::neural_network {
    template<typename T>
    class ReLU final : public ILayer<T> {
        Tensor<T, 2> mask;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            mask = x;
            return apply(x, [](T i) { return i > T(0) ? i : T(0); });
        };

        Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
            return apply(grad, [this, i = size_t{0}](T g) mutable {
                T grad = mask.begin()[i] > T(0) ? g : T(0);
                ++i;
                return grad;
            });
        };
    };

    template<typename T>
    class Sigmoid final : public ILayer<T> {
        Tensor<T, 2> mask;
    public:
        Tensor<T,2> forward(const Tensor<T,2>& z) override { 
            mask = apply(z, [](T x) {
                return T(1) / (T(1) + std::exp(-x));
            });
            return mask;
        };

        Tensor<T,2> backward(const Tensor<T,2>& g) override { 
            return apply(g, [this, i = size_t{0}](T g) mutable {
                const T& s = mask.begin()[i];
                ++i;
                return g * s * (T(1) - s);
            });
        };
    };

    template<typename T>
    class Softmax final : public ILayer<T> {
        Tensor<T, 2> mask;
    public:
        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            auto shape = x.shape();
            size_t batch = shape[0];
            size_t classes = shape[1];

            mask = Tensor<T, 2>(batch, classes);

            for (size_t i = 0; i < batch; ++i) {
                T max_val = x(i, 0);
                for (size_t j = 1; j < classes; ++j) {
                    if (x(i, j) > max_val) max_val = x(i, j);
                }

                T sum = T(0);
                for (size_t j = 0; j < classes; ++j) {
                    mask(i, j) = std::exp(x(i, j) - max_val);
                    sum += mask(i, j);
                }

                for (size_t j = 0; j < classes; ++j) {
                    mask(i, j) /= sum;
                }
            }
            return mask;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& grad) override {
            return grad;
        }
    };

    template<typename T>
    class Dropout final : public ILayer<T> {
        float p;  // dropout probability
        Tensor<T,2> mask;
        std::mt19937 gen{42};
    public:
        Dropout(float prob) : p(prob) {}
        Tensor<T,2> forward(const Tensor<T,2>& x) override {
            mask = Tensor<T,2>(x.shape());
            std::bernoulli_distribution dist(1.0f - p);
            for (size_t i = 0; i < mask.size(); ++i)
                mask.begin()[i] = dist(gen) ? T(1)/(1-p) : T(0);
            return x * mask;
        }
        Tensor<T,2> backward(const Tensor<T,2>& grad) override {
            return grad * mask;
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
