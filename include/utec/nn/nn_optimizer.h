//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include <unordered_map>
#include <cmath>

#include "nn_interfaces.h"

using utec::algebra::Tensor;

namespace utec::neural_network {
    template<typename T>
    class SGD final : public IOptimizer<T> {
        T learn_rate;
    public:
        explicit SGD(T learning_rate = 0.01) : learn_rate(learning_rate) { };
        void update(Tensor<T, 2>& params, const Tensor<T, 2>& grads) override { 
            for (size_t i = 0; i < params.size(); ++i) params.begin()[i] -= learn_rate * grads.cbegin()[i];
        };

        T& learning_rate() { return learn_rate; };
    };

    template<typename T>
    class Adam final : public IOptimizer<T> {
        T learn_rate, b1, b2, eps;
        size_t t = 0;
        std::unordered_map<const void*, Tensor<T,2>> m_map, v_map;
    public:
        explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8) : learn_rate(learning_rate), b1(beta1), b2(beta2), eps(epsilon) { };
        
        void update(Tensor<T, 2>& param, const Tensor<T, 2>& grad) override {
            ++t;
            T b1t = std::pow(b1, t), b2t = std::pow(b2, t);

            auto key = static_cast<const void*>(&param);

            if (!m_map.count(key)) {
                m_map[key] = Tensor<T,2>(param.shape());  m_map[key].fill(0);
                v_map[key] = Tensor<T,2>(param.shape());  v_map[key].fill(0);
            }
            Tensor<T,2>& m = m_map[key];
            Tensor<T,2>& v = v_map[key];

            auto it_m = m.begin(), it_v = v.begin();
            auto it_p = param.begin();
            auto it_g = grad.cbegin();
            for (; it_m != m.end(); ++it_m, ++it_v, ++it_p, ++it_g) {
                *it_m = b1 * (*it_m) + (1 - b1) * (*it_g);
                *it_v = b2 * (*it_v) + (1 - b2) * (*it_g) * (*it_g);

                T m_hat = (*it_m) / (1 - b1t);
                T v_hat = (*it_v) / (1 - b2t);

                *it_p -= learn_rate * m_hat / (std::sqrt(v_hat) + eps);
            }
        };

        T& learning_rate() { return learn_rate; };
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
