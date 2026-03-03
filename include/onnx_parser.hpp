#pragma once

#include "graph.hpp"

namespace onnx {
class ModelProto;
class GraphProto;
class NodeProto;
class ValueInfoProto;
class AttributeProto;
}  // namespace onnx

class OnnxParser {
public:
    OnnxParser() = default;
    ~OnnxParser() = default;
    OnnxParser(const OnnxParser&) = default;
    OnnxParser& operator=(const OnnxParser&) = default;
    OnnxParser(OnnxParser&&) noexcept = default;
    OnnxParser& operator=(OnnxParser&&) noexcept = default;

    Graph Parse(const onnx::ModelProto& model);

private:
    void ParseNodes(const onnx::GraphProto& proto_graph, Graph& graph);
    void ParseTensors(const onnx::GraphProto& proto_graph, Graph& graph);
    void ParseInitializers(const onnx::GraphProto& proto_graph, Graph& graph);

    Node ParseNode(const onnx::NodeProto& proto_node);
    Tensor ParseTensorInfo(const onnx::ValueInfoProto& info);
    AttrValue ParseAttribute(const onnx::AttributeProto& attr);

    DataType MapDataType(int32_t onnx_type);
};