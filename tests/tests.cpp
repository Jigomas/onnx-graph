#include <gtest/gtest.h>

#include <fstream>

#include "../include/graph.hpp"
#include "../include/graph_utils.hpp"
#include "../include/mlir_codegen.hpp"
#include "../include/node.hpp"
#include "../include/visualizer.hpp"

TEST(NodeTest, GetAttrString) {
    Node node;
    node.attributes["auto_pad"] = std::string{"SAME_UPPER"};
    EXPECT_EQ(node.GetAttr<std::string>("auto_pad"), "SAME_UPPER");
    EXPECT_EQ(node.GetAttr<std::string>("missing"), "");

    Graph graph("node_test");
    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    auto test_node = MakeNode("test_node", "Test", {"input"}, {"output"});
    graph.AddNode(std::move(test_node));

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "node_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GraphTest, FindNonexistentNodeReturnsNullptr) {
    Graph graph("empty");
    EXPECT_EQ(graph.FindNode("ghost"), nullptr);

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    auto node = MakeNode("node1", "Relu", {"input"}, {"output"});
    graph.AddNode(std::move(node));
    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "find_node_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GraphTest, AddNullNodeThrows) {
    Graph graph("test");
    EXPECT_THROW(graph.AddNode(nullptr), std::invalid_argument);

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    auto node = MakeNode("node1", "Relu", {"input"}, {"output"});
    graph.AddNode(std::move(node));
    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "add_null_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GraphTest, FindNonexistentTensorReturnsNullopt) {
    Graph graph("empty");
    auto  result = graph.FindTensor("ghost");
    EXPECT_FALSE(result.has_value());

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    auto node = MakeNode("node1", "Relu", {"input"}, {"output"});
    graph.AddNode(std::move(node));
    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "find_tensor_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GraphTest, AllOpTypesSupported) {
    Graph                    graph("ops");
    std::vector<std::string> ops = {"Add", "Mul", "Conv", "Relu", "MatMul", "Gemm"};

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    for (const auto& op : ops) {
        auto node = MakeNode(op + "_node", op, {"input"}, {"output"});
        graph.AddNode(std::move(node));
    }

    for (const auto& op : ops) {
        EXPECT_NE(graph.FindNode(op + "_node"), nullptr);
    }

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "all_ops_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(ConvTest, AllAttributes) {
    Graph graph("conv_test");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 32, 222, 222}));

    Tensor weights;
    weights.name           = "conv_weights";
    weights.dtype          = DataType::FLOAT32;
    weights.shape.dims     = {32, 3, 3, 3};
    weights.is_initializer = true;
    graph.AddTensor(weights);

    auto conv = MakeNode("conv_node", "Conv", {"input", "conv_weights"}, {"output"});
    conv->attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    conv->attributes["strides"]      = std::vector<int64_t>{1, 1};
    conv->attributes["dilations"]    = std::vector<int64_t>{1, 1};
    conv->attributes["pads"]         = std::vector<int64_t>{0, 0, 0, 0};
    conv->attributes["group"]        = int64_t{1};

    auto kernel = conv->GetAttr<std::vector<int64_t>>("kernel_shape");
    ASSERT_EQ(kernel.size(), 2);
    EXPECT_EQ(kernel[0], 3);
    EXPECT_EQ(kernel[1], 3);

    auto strides = conv->GetAttr<std::vector<int64_t>>("strides");
    ASSERT_EQ(strides.size(), 2);
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides[1], 1);

    graph.AddNode(std::move(conv));
    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "conv_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GemmTest, Attributes) {
    Graph graph("gemm_test");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    Tensor weights;
    weights.name           = "gemm_weights";
    weights.dtype          = DataType::FLOAT32;
    weights.shape.dims     = {1000, 32};
    weights.is_initializer = true;
    graph.AddTensor(weights);

    auto gemm = MakeNode("gemm_node", "Gemm", {"input", "gemm_weights"}, {"output"});
    gemm->attributes["alpha"]  = float{1.0f};
    gemm->attributes["beta"]   = float{1.0f};
    gemm->attributes["transB"] = int64_t{1};

    EXPECT_FLOAT_EQ(gemm->GetAttr<float>("alpha"), 1.0f);
    EXPECT_FLOAT_EQ(gemm->GetAttr<float>("beta"), 1.0f);
    EXPECT_EQ(gemm->GetAttr<int64_t>("transB"), 1);

    graph.AddNode(std::move(gemm));
    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "gemm_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(GraphTest, TopologicalOrder) {
    Graph graph("topo_test");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("mid", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    auto node1 = MakeNode("node1", "Conv", {"input"}, {"mid"});
    auto node2 = MakeNode("node2", "Relu", {"mid"}, {"output"});
    graph.AddNode(std::move(node1));
    graph.AddNode(std::move(node2));

    auto sorted = graph.TopologicalSort();

    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0]->name, "node1");
    EXPECT_EQ(sorted[1]->name, "node2");

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "topo_test.dot";
    viz.ToDot(graph, dot_file);
}

TEST(VisualizerTest, GeneratesDotFile) {
    Graph graph("test_viz");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    auto relu = MakeNode("relu_0", "Relu", {"input"}, {"output"});
    graph.AddNode(std::move(relu));

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "test_graph.dot";

    EXPECT_NO_THROW({ viz.ToDot(graph, dot_file); });

    std::ifstream file(dot_file);
    EXPECT_TRUE(file.good());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("relu_0"), std::string::npos);
}

TEST(AllOpsBigGraphTest, VisualizeEverything) {
    Graph graph("big_test_graph");

    graph.AddTensor(MakeTensor("input", DataType::FLOAT32, {1, 3, 224, 224}));
    graph.AddTensor(MakeTensor("conv_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("relu_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("add_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("mul_out", DataType::FLOAT32, {1, 32, 222, 222}));
    graph.AddTensor(MakeTensor("flatten_out", DataType::FLOAT32, {1, 32 * 222 * 222}));
    graph.AddTensor(MakeTensor("matmul_out", DataType::FLOAT32, {1, 1000}));
    graph.AddTensor(MakeTensor("output", DataType::FLOAT32, {1, 1000}));

    Tensor conv_weights;
    conv_weights.name           = "conv_w";
    conv_weights.dtype          = DataType::FLOAT32;
    conv_weights.shape.dims     = {32, 3, 3, 3};
    conv_weights.is_initializer = true;
    graph.AddTensor(conv_weights);

    Tensor matmul_weights;
    matmul_weights.name           = "matmul_w";
    matmul_weights.dtype          = DataType::FLOAT32;
    matmul_weights.shape.dims     = {1000, 32 * 222 * 222};
    matmul_weights.is_initializer = true;
    graph.AddTensor(matmul_weights);

    Tensor gemm_weights;
    gemm_weights.name           = "gemm_w";
    gemm_weights.dtype          = DataType::FLOAT32;
    gemm_weights.shape.dims     = {1000, 1000};
    gemm_weights.is_initializer = true;
    graph.AddTensor(gemm_weights);

    auto conv = MakeNode("conv_0", "Conv", {"input", "conv_w"}, {"conv_out"});
    conv->attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    conv->attributes["strides"]      = std::vector<int64_t>{1, 1};
    conv->attributes["dilations"]    = std::vector<int64_t>{1, 1};
    conv->attributes["pads"]         = std::vector<int64_t>{0, 0, 0, 0};
    conv->attributes["group"]        = int64_t{1};
    graph.AddNode(std::move(conv));

    graph.AddNode(MakeNode("relu_0", "Relu", {"conv_out"}, {"relu_out"}));

    auto add = MakeNode("add_0", "Add", {"relu_out", "relu_out"}, {"add_out"});
    graph.AddNode(std::move(add));

    auto mul = MakeNode("mul_0", "Mul", {"add_out", "add_out"}, {"mul_out"});
    graph.AddNode(std::move(mul));

    graph.AddNode(MakeNode("flatten_0", "Flatten", {"mul_out"}, {"flatten_out"}));

    auto matmul = MakeNode("matmul_0", "MatMul", {"flatten_out", "matmul_w"}, {"matmul_out"});
    graph.AddNode(std::move(matmul));

    auto gemm                  = MakeNode("gemm_0", "Gemm", {"matmul_out", "gemm_w"}, {"output"});
    gemm->attributes["alpha"]  = float{1.0f};
    gemm->attributes["beta"]   = float{1.0f};
    gemm->attributes["transB"] = int64_t{1};
    graph.AddNode(std::move(gemm));

    graph.inputs  = {"input"};
    graph.outputs = {"output"};

    Visualizer  viz;
    std::string dot_file = "big_test_graph.dot";
    std::string png_file = "big_test_graph.png";

    viz.ToDot(graph, dot_file);
    viz.Render(dot_file, png_file);

    std::ifstream file(dot_file);
    EXPECT_TRUE(file.good());

    std::ifstream png(png_file);
    EXPECT_TRUE(png.good());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("conv_0"), std::string::npos);
    EXPECT_NE(content.find("relu_0"), std::string::npos);
    EXPECT_NE(content.find("add_0"), std::string::npos);
    EXPECT_NE(content.find("mul_0"), std::string::npos);
    EXPECT_NE(content.find("flatten_0"), std::string::npos);
    EXPECT_NE(content.find("matmul_0"), std::string::npos);
    EXPECT_NE(content.find("gemm_0"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForAdd) {
    Graph graph("add_graph");
    graph.AddTensor(MakeTensor("a", DataType::FLOAT32, {4, 4}));
    graph.AddTensor(MakeTensor("b", DataType::FLOAT32, {4, 4}));
    graph.AddTensor(MakeTensor("c", DataType::FLOAT32, {4, 4}));
    graph.AddNode(MakeNode("add_0", "Add", {"a", "b"}, {"c"}));
    graph.inputs  = {"a", "b"};
    graph.outputs = {"c"};

    MLIRCodegen cg(graph);
    std::string mlir = cg.GenerateMLIR();

    EXPECT_NE(mlir.find("module"), std::string::npos);
    EXPECT_NE(mlir.find("func.func @model"), std::string::npos);
    EXPECT_NE(mlir.find("arith.addf"), std::string::npos);
    EXPECT_NE(mlir.find("linalg.generic"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForMul) {
    Graph graph("mul_graph");
    graph.AddTensor(MakeTensor("a", DataType::FLOAT32, {2, 8}));
    graph.AddTensor(MakeTensor("b", DataType::FLOAT32, {2, 8}));
    graph.AddTensor(MakeTensor("c", DataType::FLOAT32, {2, 8}));
    graph.AddNode(MakeNode("mul_0", "Mul", {"a", "b"}, {"c"}));
    graph.inputs  = {"a", "b"};
    graph.outputs = {"c"};

    std::string mlir = MLIRCodegen(graph).GenerateMLIR();
    EXPECT_NE(mlir.find("arith.mulf"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForRelu) {
    Graph graph("relu_graph");
    graph.AddTensor(MakeTensor("x", DataType::FLOAT32, {1, 16}));
    graph.AddTensor(MakeTensor("y", DataType::FLOAT32, {1, 16}));
    graph.AddNode(MakeNode("relu_0", "Relu", {"x"}, {"y"}));
    graph.inputs  = {"x"};
    graph.outputs = {"y"};

    std::string mlir = MLIRCodegen(graph).GenerateMLIR();
    EXPECT_NE(mlir.find("arith.maximumf"), std::string::npos);
    EXPECT_NE(mlir.find("0.000000e+00"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForMatMul) {
    Graph graph("matmul_graph");
    graph.AddTensor(MakeTensor("a", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("b", DataType::FLOAT32, {8, 4}));
    graph.AddTensor(MakeTensor("c", DataType::FLOAT32, {4, 4}));
    graph.AddNode(MakeNode("mm_0", "MatMul", {"a", "b"}, {"c"}));
    graph.inputs  = {"a", "b"};
    graph.outputs = {"c"};

    std::string mlir = MLIRCodegen(graph).GenerateMLIR();
    EXPECT_NE(mlir.find("linalg.matmul"), std::string::npos);
    EXPECT_NE(mlir.find("linalg.fill"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForGemm) {
    Graph graph("gemm_graph");
    graph.AddTensor(MakeTensor("a", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("b", DataType::FLOAT32, {4, 8}));
    graph.AddTensor(MakeTensor("c", DataType::FLOAT32, {4, 4}));

    auto gemm                  = MakeNode("gemm_0", "Gemm", {"a", "b"}, {"c"});
    gemm->attributes["transB"] = int64_t{1};
    graph.AddNode(std::move(gemm));
    graph.inputs  = {"a", "b"};
    graph.outputs = {"c"};

    std::string mlir = MLIRCodegen(graph).GenerateMLIR();
    EXPECT_NE(mlir.find("linalg.matmul_transpose_b"), std::string::npos);
}

TEST(MLIRCodegenTest, GeneratesModuleForConv) {
    Graph graph("conv_graph");
    graph.AddTensor(MakeTensor("x", DataType::FLOAT32, {1, 3, 8, 8}));
    graph.AddTensor(MakeTensor("w", DataType::FLOAT32, {8, 3, 3, 3}));
    graph.AddTensor(MakeTensor("y", DataType::FLOAT32, {1, 8, 6, 6}));

    auto conv                     = MakeNode("conv_0", "Conv", {"x", "w"}, {"y"});
    conv->attributes["strides"]   = std::vector<int64_t>{1, 1};
    conv->attributes["dilations"] = std::vector<int64_t>{1, 1};
    graph.AddNode(std::move(conv));
    graph.inputs  = {"x", "w"};
    graph.outputs = {"y"};

    std::string mlir = MLIRCodegen(graph).GenerateMLIR();
    EXPECT_NE(mlir.find("linalg.conv_2d_nchw_fchw"), std::string::npos);
    EXPECT_NE(mlir.find("dilations"), std::string::npos);
    EXPECT_NE(mlir.find("strides"), std::string::npos);
}

TEST(MLIRCodegenTest, WritesToFile) {
    Graph graph("write_test");
    graph.AddTensor(MakeTensor("a", DataType::FLOAT32, {2, 2}));
    graph.AddTensor(MakeTensor("b", DataType::FLOAT32, {2, 2}));
    graph.AddTensor(MakeTensor("c", DataType::FLOAT32, {2, 2}));
    graph.AddNode(MakeNode("add_0", "Add", {"a", "b"}, {"c"}));
    graph.inputs  = {"a", "b"};
    graph.outputs = {"c"};

    std::string path = "write_test.mlir";
    EXPECT_NO_THROW({ MLIRCodegen(graph).WriteToFile(path); });

    std::ifstream f(path);
    EXPECT_TRUE(f.good());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("module"), std::string::npos);
}