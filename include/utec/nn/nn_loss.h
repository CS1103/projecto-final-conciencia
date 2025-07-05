//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include <algorithm>
#include <cmath>

#include "nn_interfaces.h"

namespace utec::neural_network {
    template<typename T>
    class MSELoss final: public ILoss<T, 2> {
        Tensor<T, 2> last_pred, last_target;
    public:
        MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) : last_pred(y_prediction), last_target(y_true) {};

        T loss() const override {
            T sum = 0;
            auto punt_pred = last_pred.cbegin();
            auto punt_targ = last_target.cbegin();
            for (; punt_pred != last_pred.cend(); ++punt_pred, ++punt_targ)
                sum += (*punt_pred - *punt_targ) * (*punt_pred - *punt_targ);
            return sum / static_cast<T>(last_pred.size());
        };

        Tensor<T,2> loss_gradient() const override {
            T scale = T(2) / static_cast<T>(last_pred.size());
            Tensor<T, 2> grad = last_pred - last_target;
            return grad * scale;
        };
    };

    template<typename T>
    class BCELoss final: public ILoss<T, 2> {
        Tensor<T, 2> last_pred, last_target;
    public:
      BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) : last_pred(y_prediction), last_target(y_true) { };

        T loss() const override {
            T sum = 0;
            auto punt_pred = last_pred.cbegin();
            auto punt_targ = last_target.cbegin();

            for (; punt_pred != last_pred.cend(); ++punt_pred, ++punt_targ) {
                T y = *punt_targ;
                T p = std::clamp(*punt_pred, T(1e-7), T(1) - T(1e-7));
                sum += y * std::log(p) + (T(1) - y) * std::log(T(1) - p);
            }

            return -sum / static_cast<T>(last_pred.size());
        }

        Tensor<T, 2> loss_gradient() const override {
            Tensor<T, 2> grad = last_pred;
            auto punt_ante = grad.begin();
            auto punt_pred = last_pred.cbegin();
            auto punt_targ = last_target.cbegin();

            for (; punt_pred != last_pred.cend(); ++punt_pred, ++punt_targ, ++punt_ante) {
                T y = *punt_targ;
                T p = std::clamp(*punt_pred, T(1e-7), T(1) - T(1e-7));
                *punt_ante = (p - y) / (p * (T(1) - p) * static_cast<T>(last_pred.size()));
            }

            return grad;
        }
    };

    template<typename T>
    class CrossEntropyLoss final : public ILoss<T, 2> {
        Tensor<T, 2> last_pred, last_target;
    public:
        CrossEntropyLoss(const Tensor<T, 2>& y_pred, const Tensor<T, 2>& y_true)
            : last_pred(y_pred), last_target(y_true)
        {
            if (last_pred.shape()[0] != last_target.shape()[0] ||
                last_pred.shape()[1] != last_target.shape()[1])
            {
                throw std::logic_error("Mismatch between predictions and labels dimensions");
            }
        }

        T loss() const override {
            const size_t batch_size = last_pred.shape()[0];
            const size_t num_classes = last_pred.shape()[1];
            T total = 0;

            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < num_classes; ++j) {
                    // only accumulate for the true class (y_true==1)
                    T y = last_target(i, j);
                    if (y > T(0)) {
                        T p = std::clamp(last_pred(i, j), T(1e-7), T(1));
                        total -= y * std::log(p);
                    }
                }
            }
            return total / static_cast<T>(batch_size);
        }

        Tensor<T, 2> loss_gradient() const override {
            const size_t batch_size = last_pred.shape()[0];
            // gradient is (probs - y_true) / batch_size
            Tensor<T, 2> grad = last_pred - last_target;
            return grad / static_cast<T>(batch_size);
        }
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
