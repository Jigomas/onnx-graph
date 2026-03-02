#include <gtest/gtest.h>
#include <fstream>

#include "../include/graph.hpp"
#include "../include/node.hpp"
#include "../include/graph_utils.hpp"
#include "../include/onnx_loader.hpp"
#include "../include/visualizer.hpp"



TEST(NodeTest, GetAttrString) {
    Node node;
    node.attributes["auto_pad"] = std::string{"SAME_UPPER"};
    EXPECT_EQ(node.GetAttr<std::string>("auto_pad"), "SAME_UPPER");
    EXPECT_EQ(node.GetAttr<std::string>("missing"), "");
}



TEST(GraphTest, FindNonexistentNodeReturnsNullptr) {
    Graph graph("empty");
    EXPECT_EQ(graph.FindNode("ghost"), nullptr);
}



TEST(GraphTest, AddNullNodeThrows) {
    Graph graph("test");
    EXPECT_THROW(graph.AddNode(nullptr), std::invalid_argument);
}



TEST(GraphTest, FindNonexistentTensorReturnsNullopt) {
    Graph graph("empty");
    auto result = graph.FindTensor("ghost");
    EXPECT_FALSE(result.has_value());
}



TEST(GraphTest, AllOpTypesSupported) {
    Graph graph("ops");
    std::vector<std::string> ops = {"Add", "Mul", "Conv", "Relu", "MatMul", "Gemm"};
    
    for (const auto& op : ops) {
        auto node = MakeNode(op + "_node", op, {}, {});
        graph.AddNode(std::move(node));
    }
    
    for (const auto& op : ops) {
        EXPECT_NE(graph.FindNode(op + "_node"), nullptr);
    }
}



TEST(ConvTest, AllAttributes) {
    Node conv;
    conv.op_type = "Conv";
    conv.attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    conv.attributes["strides"] = std::vector<int64_t>{1, 1};
    conv.attributes["dilations"] = std::vector<int64_t>{1, 1};
    conv.attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    conv.attributes["group"] = int64_t{1};
    
    auto kernel = conv.GetAttr<std::vector<int64_t>>("kernel_shape");
    ASSERT_EQ(kernel.size(), 2);
    EXPECT_EQ(kernel[0], 3);
    EXPECT_EQ(kernel[1], 3);
    
    auto strides = conv.GetAttr<std::vector<int64_t>>("strides");
    ASSERT_EQ(strides.size(), 2);
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides[1], 1);
    
    auto dilations = conv.GetAttr<std::vector<int64_t>>("dilations");
    ASSERT_EQ(dilations.size(), 2);
    EXPECT_EQ(dilations[0], 1);
    EXPECT_EQ(dilations[1], 1);
    
    auto pads = conv.GetAttr<std::vector<int64_t>>("pads");
    ASSERT_EQ(pads.size(), 4);
    EXPECT_EQ(pads[0], 0);
    EXPECT_EQ(pads[1], 0);
    EXPECT_EQ(pads[2], 0);
    EXPECT_EQ(pads[3], 0);
    
    auto group = conv.GetAttr<int64_t>("group");
    EXPECT_EQ(group, 1);
}



TEST(GemmTest, Attributes) {
    Node gemm;
    gemm.op_type = "Gemm";
    gemm.attributes["alpha"] = float{1.0f};
    gemm.attributes["beta"] = float{1.0f};
    gemm.attributes["transB"] = int64_t{1};
    
    EXPECT_FLOAT_EQ(gemm.GetAttr<float>("alpha"), 1.0f);
    EXPECT_FLOAT_EQ(gemm.GetAttr<float>("beta"), 1.0f);
    EXPECT_EQ(gemm.GetAttr<int64_t>("transB"), 1);
}



TEST(GraphTest, TopologicalOrder) {
    Graph graph("test");
    
    auto node1 = MakeNode("node1", "Relu", {"input"}, {"middle"});
    auto node2 = MakeNode("node2", "Relu", {"middle"}, {"output"});
    graph.AddNode(std::move(node1));
    graph.AddNode(std::move(node2));
    
    auto sorted = graph.TopologicalSort();
    
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0]->name, "node1");
    EXPECT_EQ(sorted[1]->name, "node2");
}



TEST(VisualizerTest, GeneratesDotFile) {
    Graph graph("test_viz");
    
    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));
    
    auto relu = MakeNode("relu_0", "Relu", {"input"}, {"output"});
    graph.AddNode(std::move(relu));
    
    graph.inputs = {"input"};
    graph.outputs = {"output"};
    
    Visualizer viz;
    std::string dot_file = "test_graph.dot";
    
    EXPECT_NO_THROW({
        viz.ToDot(graph, dot_file);
    });
    
    std::ifstream file(dot_file);
    EXPECT_TRUE(file.good());
    
    std::string content((std::istreambuf_iterator<char>(file)), 
                         std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("relu_0"), std::string::npos);
    
    std::remove(dot_file.c_str());
}
