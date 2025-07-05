//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"

using namespace utec::algebra;

namespace utec::neural_network {
    template<typename T>
    class Dense : public ILayer<T> {
        size_t elm_in, elm_out;
        Tensor<T, 2> W, dW;
        Tensor<T, 2> b, db;
        Tensor<T, 2> last_x;
    public:
        template<typename InitWFun, typename InitBFun>
        Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) : W(out_f, in_f), dW(out_f, in_f), b(1, out_f), db(1, out_f), elm_in(in_f), elm_out(out_f) {
            init_w_fun(W);
            init_b_fun(b);
        };

        Tensor<T, 2> forward(const Tensor<T, 2>& x) override {
            last_x = x;
            auto out = matrix_product(x, W.transpose_2d());
            for (size_t i = 0; i < out.shape()[0]; i++){
                for (size_t j = 0; j < out.shape()[1]; j++) {
                    out(i, j) += b(0, j);
                }
            }
            return out;
        }

        Tensor<T, 2> backward(const Tensor<T, 2>& dZ) override {
            dW = matrix_product(dZ.transpose_2d(), last_x);
            db = Tensor<T, 2>(1, elm_out);
            db.fill(T(0));
            for (size_t j = 0; j < elm_out; j++)
                for (size_t i = 0; i < dZ.shape()[0]; i++) db(0, j) += dZ(i, j);

            return matrix_product(dZ, W);
        }

        void update_params(IOptimizer<T>& optimizer) override {
            optimizer.update(W,dW);
            optimizer.update(b,db);
        };

        Tensor<T, 2>& weight() { return W; }
        Tensor<T, 2>& bias() { return b; }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
