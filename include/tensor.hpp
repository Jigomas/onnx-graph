#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class DataType { FLOAT32, INT32, INT64, FLOAT64, UNKNOWN };

class TensorShape {
public:
    std::vector<int64_t> dims;
    bool is_dynamic = false;

    int64_t rank() const noexcept { return static_cast<int64_t>(dims.size()); }

    int64_t NumElements() const noexcept {
        if (dims.empty())
            return 0;
        int64_t total = 1;
        for (int64_t d : dims) {
            if (d < 0)
                return -1;
            total *= d;
        }
        return total;
    }

    bool IsScalar() const noexcept { return dims.empty(); }

    TensorShape() = default;
    ~TensorShape() = default;
    TensorShape(const TensorShape&) = default;
    TensorShape& operator=(const TensorShape&) = default;
    TensorShape(TensorShape&&) noexcept = default;
    TensorShape& operator=(TensorShape&&) noexcept = default;
};

class Tensor {
public:
    std::string name;
    DataType dtype = DataType::UNKNOWN;
    TensorShape shape;
    bool is_initializer = false;
    std::vector<float> data;

    Tensor() = default;
    ~Tensor() = default;

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
};