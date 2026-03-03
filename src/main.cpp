#include <iostream>

#include "../include/graph.hpp"
#include "../include/graph_utils.hpp"
#include "../include/node.hpp"
#include "../include/onnx_loader.hpp"
#include "../include/tensor.hpp"
#include "../include/visualizer.hpp"

static Graph BuildManualGraph() {
    Graph graph("graph");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("conv_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("relu_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    Tensor conv_weights;
    conv_weights.name = "conv_w";
    conv_weights.dtype = DataType::FLOAT32;
    conv_weights.shape.dims = {32, 3, 3, 3};
    conv_weights.is_initializer = true;
    graph.AddTensor(conv_weights);

    Tensor gemm_weights;
    gemm_weights.name = "gemm_w";
    gemm_weights.dtype = DataType::FLOAT32;
    gemm_weights.shape.dims = {1000, 32};
    gemm_weights.is_initializer = true;
    graph.AddTensor(gemm_weights);

    auto conv = MakeNode("conv_0", "Conv", {"input", "conv_w"}, {"conv_out"});
    conv->attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    conv->attributes["strides"] = std::vector<int64_t>{1, 1};
    conv->attributes["dilations"] = std::vector<int64_t>{1, 1};
    conv->attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    conv->attributes["group"] = int64_t{1};
    graph.AddNode(std::move(conv));

    graph.AddNode(MakeNode("relu_0", "Relu", {"conv_out"}, {"relu_out"}));

    auto gemm = MakeNode("gemm_0", "Gemm", {"relu_out", "gemm_w"}, {"output"});
    gemm->attributes["alpha"] = float{1.0f};
    gemm->attributes["beta"] = float{1.0f};
    gemm->attributes["transB"] = int64_t{1};
    graph.AddNode(std::move(gemm));

    graph.inputs = {"input"};
    graph.outputs = {"output"};

    return graph;
}

int main(int argc, char* argv[]) {
    try {
        Graph graph = (argc >= 2) ? OnnxLoader().Load(argv[1]) : BuildManualGraph();

        graph.DumpGraph();

        std::cout << "\n Топологический обход \n";
        for (const Node* node : graph.TopologicalSort())
            std::cout << "  [" << node->op_type << "] " << node->name << '\n';

        const Node* conv_node = graph.FindNode("conv_0");
        if (conv_node) {
            std::cout << "\n Атрибуты Conv \n";
            auto strides = conv_node->GetAttr<std::vector<int64_t>>("strides");
            auto group = conv_node->GetAttr<int64_t>("group", 1);
            std::cout << "  strides: [" << strides[0] << ", " << strides[1] << "]\n";
            std::cout << "  group:   " << group << '\n';
        }

        const Node* gemm_node = graph.FindNode("gemm_0");
        if (gemm_node) {
            std::cout << "\n Атрибуты Gemm \n";
            auto alpha = gemm_node->GetAttr<float>("alpha", 1.0f);
            auto transB = gemm_node->GetAttr<int64_t>("transB", 0);
            std::cout << "  alpha:  " << alpha << '\n';
            std::cout << "  transB: " << transB << '\n';
        }

        if (argc >= 3) {
            Visualizer viz;
            viz.ToDot(graph, "output.dot");
            viz.Render("output.dot", argv[2]);
            std::cout << "\nSaved: " << argv[2] << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}