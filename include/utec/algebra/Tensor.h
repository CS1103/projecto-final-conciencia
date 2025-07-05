//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <cstddef>
#include <iostream>
#include <array>
#include <vector>

namespace utec::algebra {
    template<typename T, size_t Tier>
    class Tensor {
        std::array<size_t, Tier> forma;
        std::array<size_t, Tier> subcorte;
        std::vector<T> data;
    public:
        Tensor() {
            forma.fill(0);
            subcorte.fill(1);
        }

        template<typename... Dims> 
        Tensor(Dims ...dims) {
            std::vector<size_t> dims_out{ static_cast<size_t>(dims)... };

            if (dims_out.size() != Tier) { // hubiera puesto un sizeof...() pero seria redundante
                std::string mal = "Number of dimensions do not match with " + std::to_string(Tier);
                throw std::logic_error(mal);
            }

            for (size_t i = 0; i < Tier; i++) forma[i] = dims_out[i];

            subcorte[Tier - 1] = 1;
            for (int i = (int)Tier - 2; i >= 0; i--) subcorte[i] = forma[i + 1] * subcorte[i + 1]; // solo corre si el tier >= 2

            size_t tam_total = 1;
            for (auto dim : forma) tam_total *= dim;
            data.resize(tam_total);
        };
        
        Tensor(const std::array<size_t, Tier>& shape) {
            forma = shape;

            subcorte[Tier - 1] = 1;
            for (int i = (int)Tier - 2; i >= 0; i--) subcorte[i] = forma[i + 1] * subcorte[i + 1];

            size_t tam_total = 1;
            for (auto dim : forma) tam_total *= dim;
            data.resize(tam_total);
        };

        template<typename... Idxs>
        T& operator()(Idxs... idxs) {
            std::array<size_t, Tier> indices{ static_cast<size_t>(idxs)... };

            size_t ind_plano = 0;
            for (size_t i = 0; i < Tier; i++) {
                if (indices[i] >= forma[i]) throw std::out_of_range("Tensor index out of range");
                ind_plano += indices[i] * subcorte[i];
            }
        
            return data[ind_plano];
        };

        template<typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            std::array<size_t, Tier> indices{ static_cast<size_t>(idxs)... };

            size_t ind_plano = 0;
            for (size_t i = 0; i < Tier; i++) {
                if (indices[i] >= forma[i]) throw std::out_of_range("Tensor index out of range");
                ind_plano += indices[i] * subcorte[i];
            }
        
            return data[ind_plano];
        };
        
        const std::array<size_t, Tier>& shape() const noexcept { return forma; };
        std::vector<T>& get_data() const noexcept { return data; };

        void reshape(const std::array<size_t, Tier>& new_shape) {
            size_t total = 1;
            for (auto dim : new_shape) total *= dim;

            if (total != data.size()) data.resize(total);

            forma = new_shape;
            subcorte[Tier - 1] = 1;
            for (int i = (int)Tier - 2; i >= 0; i--) subcorte[i] = forma[i + 1] * subcorte[i + 1];
        };

        template<typename... Dims>
        void reshape(Dims... dims) {
            std::vector<size_t> dims_out{ static_cast<size_t>(dims)... };

            if (dims_out.size() != Tier) {
                std::string mal = "Number of dimensions do not match with " + std::to_string(Tier);
                throw std::logic_error(mal.c_str());
            }

            std::array<size_t, Tier> out;
            for (size_t i = 0; i < Tier; i++) out[i] = dims_out[i];
            reshape(out);
        }

        void fill(const T& value) noexcept {
            std::fill(data.begin(), data.end(), value);
        };

        std::vector<T>::iterator begin() { return data.begin(); };
        std::vector<T>::iterator end() { return data.end(); };

        std::vector<T>::const_iterator cbegin() const { return data.cbegin(); };
        std::vector<T>::const_iterator cend() const { return data.cend(); };

        size_t size() const { return data.size(); };

        Tensor operator+(const Tensor<T, Tier>& other) const {
            for (size_t i = 0; i < Tier; i++) {
                if (forma[i] != other.forma[i] && forma[i] != 1 && other.forma[i] != 1) {
                    throw std::logic_error("Shapes do not match and they are not compatible for broadcasting");
                }
            }

            std::array<size_t, Tier> out_shape;
            for (size_t i = 0; i < Tier; i++) out_shape[i] = std::max(forma[i], other.forma[i]);

            Tensor<T, Tier> result(out_shape);

            size_t total = 1;
            for (size_t dim : out_shape) total *= dim;

            std::array<size_t, Tier> index;
            for (size_t i = 0; i < total; i++) {
                size_t rem = i, flat_a = 0, flat_b = 0;
                for (int j = (int)Tier - 1; j >= 0; j--) {
                    index[j] = rem % out_shape[j];
                    rem = rem / out_shape[j];
                    size_t idxA = (forma[j] == 1) ? 0 : index[j];
                    size_t idxB = (other.forma[j] == 1) ? 0 : index[j];
                    flat_a += idxA * subcorte[j];
                    flat_b += idxB * other.subcorte[j];
                }
                result.data[i] = data[flat_a] + other.data[flat_b];
            }
            return result;
        }

        Tensor operator-(const Tensor<T, Tier>& other) const {
            for (size_t i = 0; i < Tier; i++) {
                if (forma[i] != other.forma[i] && forma[i] != 1 && other.forma[i] != 1) {
                    throw std::logic_error("Shapes do not match and they are not compatible for broadcasting");
                }
            }

            std::array<size_t, Tier> out_shape;
            for (size_t i = 0; i < Tier; i++) {
                out_shape[i] = std::max(forma[i], other.forma[i]);
            }

            Tensor<T, Tier> result(out_shape);

            size_t total = 1;
            for (size_t dim : out_shape) {
                total *= dim;
            }

            std::array<size_t, Tier> index;
            for (size_t i = 0; i < total; i++) {
                size_t rem = i;
                size_t flat_a = 0, flat_b = 0;
                for (int j = (int)Tier - 1; j >= 0; j--) {
                    index[j] = rem % out_shape[j];
                    rem = rem / out_shape[j];
                    size_t idxA = (forma[j] == 1) ? 0 : index[j];
                    size_t idxB = (other.forma[j] == 1) ? 0 : index[j];
                    flat_a += idxA * subcorte[j];
                    flat_b += idxB * other.subcorte[j];
                }
                result.data[i] = data[flat_a] - other.data[flat_b];
            }
            return result;
        };

        Tensor operator*(const Tensor<T, Tier>& other) const {
            for (size_t i = 0; i < Tier; i++) {
                if (forma[i] != other.forma[i] && forma[i] != 1 && other.forma[i] != 1) {
                    throw std::logic_error("Shapes do not match and they are not compatible for broadcasting");
                }
            }

            std::array<size_t, Tier> out_shape;
            for (size_t i = 0; i < Tier; i++) {
                out_shape[i] = std::max(forma[i], other.forma[i]);
            }

            Tensor<T, Tier> result(out_shape);

            size_t total = 1;
            for (size_t dim : out_shape) total *= dim;

            std::array<size_t, Tier> index;
            for (size_t i = 0; i < total; i++) {
                size_t rem = i;
                size_t flat_a = 0, flat_b = 0;
                for (int j = (int)Tier - 1; j >= 0; j--) {
                    index[j] = rem % out_shape[j];
                    rem = rem / out_shape[j];
                    size_t idxA = (forma[j] == 1) ? 0 : index[j];
                    size_t idxB = (other.forma[j] == 1) ? 0 : index[j];
                    flat_a += idxA * subcorte[j];
                    flat_b += idxB * other.subcorte[j];
                }
                result.data[i] = data[flat_a] * other.data[flat_b];
            }
            return result;
        };

        // ^^^
        // si el profesor ve esto: copie el codigo 3 veces pero cambiando el operator, porque la logica deberia de ser la misma

        Tensor operator+(const T& change) const {
            Tensor<T, Tier> res = *this;
            for (auto &elemento : res.data) elemento += change;
            return res;
        };

        Tensor operator-(const T& change) const {
            Tensor<T, Tier> res = *this;
            for (auto &elemento : res.data) elemento -= change;
            return res;
        };

        Tensor operator*(const T& scalar) const {
            Tensor<T, Tier> res = *this;
            for (auto &elemento : res.data) elemento *= scalar;
            return res;
        };

        Tensor operator/(const T& scalar) const {
            Tensor<T, Tier> res = *this;
            for (auto &elemento : res.data) elemento /= scalar;
            return res;
        };

        // ^^^
        // aqui tambien

        Tensor& operator=(std::initializer_list<T> values) {
            if (values.size() != data.size()) throw std::logic_error("Data size does not match tensor size");
            std::copy(values.begin(), values.end(), data.begin());
            return *this;
        }

        // solo añadi esto porque hay un test que lo pide

        Tensor transpose_2d() const {
            if constexpr (Tier < 2) throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        
            Tensor<T, Tier> res(forma);
            std::swap(res.forma[Tier - 2], res.forma[Tier - 1]);

            res.subcorte[Tier - 1] = 1;
            for (int i = static_cast<int>(Tier) - 2; i >= 0; i--) {
                res.subcorte[i] = res.subcorte[i + 1] * res.forma[i + 1];
            }

            res.data.resize(data.size());
        
            std::vector<size_t> idx(Tier);
            for (size_t flat = 0; flat < data.size(); ++flat) {
                size_t residual = flat;
                for (size_t i = 0; i < Tier; ++i) {
                    idx[i] = residual / subcorte[i];
                    residual = residual % subcorte[i];
                }
        
                std::swap(idx[Tier - 2], idx[Tier - 1]);

                size_t new_flat = 0;
                for (size_t i = 0; i < Tier; ++i) new_flat += idx[i] * res.subcorte[i];
        
                res.data[new_flat] = data[flat];
            }
        
            return res;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
            const auto& la_forma = tensor.shape();
            auto dato_ten = tensor.cbegin();
    
            if constexpr (Tier == 1) {
                for (size_t i = 0; i < la_forma[0]; i++, dato_ten++) {
                    os << *dato_ten;
                    if (i + 1 < la_forma[0]) os << " ";
                }
            }
            else if constexpr (Tier == 2) {
                size_t rows = la_forma[0];
                size_t cols = la_forma[1];
                os << "{\n";
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++, dato_ten++) {
                        os << *dato_ten;
                        if (j + 1 < cols) os << " ";
                    }
                    os << "\n";
                }
                os << "}\n";
            }
            else {
                auto slice_size = [](const std::array<size_t, Tier>& shape, size_t start) {
                    size_t size = 1;
                    for (size_t i = start; i < Tier; ++i) size *= shape[i];
                    return size;
                };
        
                auto imp_recursiva = [&](auto&& self, size_t level, size_t indent, typename std::vector<T>::const_iterator it) -> void {
                    os << std::string(indent, ' ') << "{\n";
        
                    if (level == Tier - 2) {
                        size_t rows = la_forma[level];
                        size_t cols = la_forma[level + 1];
                        for (size_t i = 0; i < rows; ++i) {
                            os << std::string(indent + 2, ' ');
                            for (size_t j = 0; j < cols; ++j, ++it) {
                                os << *it;
                                if (j + 1 < cols) os << " ";
                            }
                            os << "\n";
                        }
                    } else {
                        size_t step = slice_size(la_forma, level + 1);
                        for (size_t i = 0; i < la_forma[level]; ++i) {
                            self(self, level + 1, indent + 2, it);
                            it += step;
                        }
                    }
        
                    os << std::string(indent, ' ') << "}\n";
                };
        
                imp_recursiva(imp_recursiva, 0, 0, dato_ten);
            }
            return os;
        };

        template<typename T1, size_t R2>
        friend Tensor<T1, R2> matrix_product(const Tensor<T1, R2> &t1, const Tensor<T1, R2> &t2);

        template<typename T1, size_t R2, typename Func>
        friend Tensor<T1, R2> apply(const Tensor<T1, R2> &tensor, Func f);
    };

    template<typename T, size_t Tier>
    Tensor<T, Tier> operator+(const T& change, const Tensor<T, Tier>& tensor) {
        return tensor + change; // vamos a usar los operadores q ya definimos y para las demas tmb
    }

    template<typename T, size_t Tier>
    Tensor<T, Tier> operator-(const T& change, const Tensor<T, Tier>& tensor) {
        return tensor - change;
    }

    template<typename T, size_t Tier>
    Tensor<T, Tier> operator*(const T& scalar, const Tensor<T, Tier>& tensor) {
        return tensor * scalar;
    }

    template<typename T, size_t Tier>
    Tensor<T, Tier> operator/(const T& scalar, const Tensor<T, Tier>& tensor) {
        return tensor / scalar;
    }

    template<typename T, size_t Tier>
    Tensor<T, Tier> transpose_2d(Tensor<T, Tier> tensor) {
        return tensor.transpose_2d();
    }

    template<typename T, size_t Tier>
    Tensor<T, Tier> matrix_product(const Tensor<T, Tier> &t1, const Tensor<T, Tier> &t2){
        const auto& pieza1 = t1.forma;
        const auto& pieza2 = t2.forma;
    
        if (pieza1[Tier - 1] != pieza2[Tier - 2]) {
            throw std::logic_error("Matrix dimensions are incompatible for multiplication");
        }

        for (size_t i = 0; i < Tier - 2; i++) {
            if (pieza1[i] != pieza2[i]) {
                throw std::logic_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }

        size_t row = pieza1[Tier-2], col1 = pieza1[Tier-1], col2 = pieza2[Tier-1];
        std::array<size_t, Tier> out_shape = pieza1;
        out_shape[Tier-1] = col2;
        Tensor<T, Tier> res(out_shape);
        res.data.assign(res.data.size(), T(0));

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < row; ++i) {
            for (size_t j = 0; j < col2; ++j) {
                T sum = 0;
                for (size_t k = 0; k < col1; ++k) {
                    // calculamos los cortes manualmente
                    size_t idx1 = i * t1.subcorte[Tier-2] + k * t1.subcorte[Tier-1];
                    size_t idx2 = k * t2.subcorte[Tier-2] + j * t2.subcorte[Tier-1];
                    sum += t1.data[idx1] * t2.data[idx2];
                }
                size_t out_idx = i * res.subcorte[Tier-2] + j * res.subcorte[Tier-1];
                res.data[out_idx] = sum;
            }
        }
        return res;
    }

    template<typename T, size_t Tier, typename Func>
    Tensor<T, Tier> apply(const Tensor<T, Tier>& tensor, Func f) {
        Tensor<T, Tier> result = tensor;
        size_t total = result.data.size();
        #pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            result.data[idx] = f(tensor.data[idx]);
        }
        return result;
    }

}

namespace utec {
    template<typename T, size_t DIMS>
    using Tensor = algebra::Tensor<T, DIMS>;
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
