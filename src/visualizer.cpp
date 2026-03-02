#include "../include/visualizer.hpp"

#include <fstream>
#include <stdexcept>
#include <cstdlib>



void Visualizer::ToDot(const Graph& graph, const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) throw std::runtime_error("Cannot write dot file: " + filepath);

    file << "digraph \"" << EscapeLabel(graph.name) << "\" {\n";
    file << "  rankdir=TB;\n";
    file << "  node [shape=box, style=filled, fontname=Helvetica];\n\n";

    // Узлы
    for (const auto& node_ptr : graph.nodes) {
        file << "  \"" << node_ptr->name << "\" [label=\""
             << node_ptr->op_type << "\\n" << node_ptr->name
             << "\", fillcolor=" << GetNodeColor(node_ptr->op_type) << "];\n";
    }

    file << '\n';

    for (const auto& node_ptr : graph.nodes) {
        for (const auto& input_name : node_ptr->inputs) {
            for (const auto& producer_ptr : graph.nodes) {
                for (const auto& out : producer_ptr->outputs) {
                    if (out == input_name) {
                        file << "  \"" << producer_ptr->name
                             << "\" -> \""
                             << node_ptr->name
                             << "\" [label=\"" << input_name << "\"];\n";
                    }
                }
            }
        }
    }

    file << "}\n";
}



void Visualizer::Render(const std::string& dot_file, const std::string& out_png) const {
    std::string cmd = "dot -Tpng " + dot_file + " -o " + out_png;
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("Graphviz render failed. Install: sudo apt install graphviz");
}



std::string Visualizer::GetNodeColor(const std::string& op_type) const {
    if (op_type == "Conv")               return "lightblue";
    if (op_type == "Relu")               return "lightgreen";
    if (op_type == "Sigmoid")            return "lightgreen";
    if (op_type == "Tanh")               return "lightgreen";
    if (op_type == "Gemm")               return "lightyellow";
    if (op_type == "MatMul")             return "lightyellow";
    if (op_type == "Add")                return "lightsalmon";
    if (op_type == "Mul")                return "lightsalmon";
    if (op_type == "BatchNormalization") return "plum";
    if (op_type == "MaxPool")            return "lightcyan";
    if (op_type == "Flatten")            return "lightcyan";
    return "white";
}



std::string Visualizer::EscapeLabel(const std::string& label) const {
    std::string result;
    result.reserve(label.size());
    for (char c : label) {
        if (c == '"' || c == '\\') result += '\\';
        result += c;
    }
    return result;
}