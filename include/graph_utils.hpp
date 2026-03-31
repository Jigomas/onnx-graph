#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "node.hpp"
#include "tensor.hpp"

inline std::unique_ptr<Node> MakeNode(const std::string&              name,
                                      const std::string&              op_type,
                                      const std::vector<std::string>& inputs,
                                      const std::vector<std::string>& outputs) {
    if (op_type.empty())
        throw std::invalid_argument("MakeNode: op_type must not be empty");

    auto node     = std::make_unique<Node>();
    node->name    = name;
    node->op_type = op_type;
    node->inputs  = inputs;
    node->outputs = outputs;
    return node;
}

inline Tensor MakeTensor(const std::string& name, DataType dtype, std::vector<int64_t> dims) {
    Tensor tensor;
    tensor.name       = name;
    tensor.dtype      = dtype;
    tensor.shape.dims = std::move(dims);

    return tensor;
}