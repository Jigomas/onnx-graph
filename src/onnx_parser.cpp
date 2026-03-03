#include "../include/onnx_parser.hpp"

#include <cstring>

#include "onnx.pb.h"

Graph OnnxParser::Parse(const onnx::ModelProto& model) {
    Graph graph(model.graph().name());
    const auto& pg = model.graph();

    ParseTensors(pg, graph);
    ParseInitializers(pg, graph);
    ParseNodes(pg, graph);

    for (const auto& inp : pg.input())
        graph.inputs.push_back(inp.name());
    for (const auto& out : pg.output())
        graph.outputs.push_back(out.name());

    return graph;
}

void OnnxParser::ParseNodes(const onnx::GraphProto& proto_graph, Graph& graph) {
    for (const auto& proto_node : proto_graph.node())
        graph.AddNode(std::make_unique<Node>(ParseNode(proto_node)));
}

void OnnxParser::ParseTensors(const onnx::GraphProto& proto_graph, Graph& graph) {
    for (const auto& info : proto_graph.value_info())
        graph.AddTensor(ParseTensorInfo(info));
}

void OnnxParser::ParseInitializers(const onnx::GraphProto& proto_graph, Graph& graph) {
    for (const auto& init : proto_graph.initializer()) {
        Tensor tensor;
        tensor.name = init.name();
        tensor.dtype = MapDataType(init.data_type());
        tensor.is_initializer = true;

        for (int64_t dim : init.dims())
            tensor.shape.dims.push_back(dim);

        if (init.float_data_size() > 0) {
            tensor.data.assign(init.float_data().begin(), init.float_data().end());
        } else if (!init.raw_data().empty() && tensor.dtype == DataType::FLOAT32) {
            const size_t byte_count = init.raw_data().size();

            if (byte_count % sizeof(float) != 0)
                throw std::runtime_error(
                    "OnnxParser::ParseInitializers: raw_data size " + std::to_string(byte_count) +
                    " is not divisible by sizeof(float) for initializer '" + init.name() + "'");

            const size_t count = byte_count / sizeof(float);
            tensor.data.resize(count);
            std::memcpy(tensor.data.data(), init.raw_data().data(), byte_count);
        }

        graph.AddTensor(std::move(tensor));
    }
}

Node OnnxParser::ParseNode(const onnx::NodeProto& proto_node) {
    Node node;
    node.name = proto_node.name();
    node.op_type = proto_node.op_type();  // "Conv", "Relu", "Add"...

    for (const auto& inp : proto_node.input())
        node.inputs.push_back(inp);
    for (const auto& out : proto_node.output())
        node.outputs.push_back(out);

    for (const auto& attr : proto_node.attribute())
        node.attributes[attr.name()] = ParseAttribute(attr);

    return node;
}

Tensor OnnxParser::ParseTensorInfo(const onnx::ValueInfoProto& info) {
    Tensor tensor;
    tensor.name = info.name();

    if (!info.type().has_tensor_type())
        return tensor;

    const auto& tt = info.type().tensor_type();
    tensor.dtype = MapDataType(tt.elem_type());

    if (tt.has_shape()) {
        for (const auto& dim : tt.shape().dim()) {
            if (dim.has_dim_value()) {
                tensor.shape.dims.push_back(dim.dim_value());
            } else {
                tensor.shape.dims.push_back(-1);
                tensor.shape.is_dynamic = true;
            }
        }
    }

    return tensor;
}

AttrValue OnnxParser::ParseAttribute(const onnx::AttributeProto& attr) {
    switch (attr.type()) {
        case onnx::AttributeProto::INT:
            return attr.i();

        case onnx::AttributeProto::FLOAT:
            return attr.f();

        case onnx::AttributeProto::STRING:
            return attr.s();

        case onnx::AttributeProto::INTS:
            return std::vector<int64_t>(attr.ints().begin(), attr.ints().end());

        case onnx::AttributeProto::FLOATS:
            return std::vector<float>(attr.floats().begin(), attr.floats().end());

        default:
            throw std::runtime_error("OnnxParser::ParseAttribute: unsupported attribute type=" +
                                     std::to_string(static_cast<int>(attr.type())) +
                                     " for attribute '" + attr.name() + "'");
    }
}

DataType OnnxParser::MapDataType(int32_t onnx_type) {
    // onnx.proto3: 1=FLOAT, 6=INT32, 7=INT64, 11=DOUBLE
    switch (onnx_type) {
        case 1:
            return DataType::FLOAT32;
        case 6:
            return DataType::INT32;
        case 7:
            return DataType::INT64;
        case 11:
            return DataType::FLOAT64;
        default:
            return DataType::UNKNOWN;
    }
}