#include "../include/visualizer.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

void Visualizer::ToDot(const Graph& graph, const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file)
        throw std::runtime_error("Cannot write dot file: " + filepath);

    file << "digraph \"" << EscapeLabel(graph.name) << "\" {\n";
    file << "  rankdir=TB;\n";
    file << "  node [shape=box, style=filled, fontname=Helvetica];\n\n";

    // Узлы
    for (const auto& node_ptr : graph.nodes) {
        file << "  \"" << node_ptr->name << "\" [label=\"" << node_ptr->op_type << "\\n"
             << node_ptr->name << "\", fillcolor=" << GetNodeColor(node_ptr->op_type) << "];\n";
    }

    file << '\n';

    std::unordered_map<std::string, std::string> tensor_producer;
    for (const auto& node_ptr : graph.nodes) {
        for (const auto& out : node_ptr->outputs) {
            tensor_producer[out] = node_ptr->name;
        }
    }

    for (const auto& node_ptr : graph.nodes) {
        for (const auto& input_name : node_ptr->inputs) {
            auto it = tensor_producer.find(input_name);
            if (it == tensor_producer.end())
                continue;

            file << "  \"" << it->second << "\" -> \"" << node_ptr->name << "\" [label=\""
                 << EscapeLabel(input_name) << "\"];\n";
        }
    }

    file << "}\n";

    if (!file.good())
        throw std::runtime_error("Visualizer::ToDot: write error for file '" + filepath + "'");
}

void Visualizer::Render(const std::string& dot_file, const std::string& out_png) const {
    // no shell-metasymbols
    auto is_safe_path = [](const std::string& p) {
        for (char c : p) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '_' && c != '-' &&
                c != '/')
                return false;
        }
        return !p.empty();
    };

    if (!is_safe_path(dot_file))
        throw std::invalid_argument("Visualizer::Render: unsafe characters in dot_file path: '" +
                                    dot_file + "'");
    if (!is_safe_path(out_png))
        throw std::invalid_argument("Visualizer::Render: unsafe characters in out_png path: '" +
                                    out_png + "'");

    const std::string cmd = "dot -Tpng " + dot_file + " -o " + out_png;
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("Visualizer::Render: graphviz failed for '" + dot_file +
                                 "'. Install: sudo apt install graphviz");
}

std::string Visualizer::GetNodeColor(const std::string& op_type) const {
    static const std::unordered_map<std::string, std::string> kColorMap = {
        {"Conv", "lightblue"},
        {"Relu", "lightgreen"},
        {"Sigmoid", "lightgreen"},
        {"Tanh", "lightgreen"},
        {"Gemm", "lightyellow"},
        {"MatMul", "lightyellow"},
        {"Add", "lightsalmon"},
        {"Mul", "lightsalmon"},
        {"BatchNormalization", "plum"},
        {"MaxPool", "lightcyan"},
        {"Flatten", "lightcyan"},
    };

    auto it = kColorMap.find(op_type);
    return (it != kColorMap.end()) ? it->second : "white";
}

std::string Visualizer::EscapeLabel(const std::string& label) const {
    std::string result;
    result.reserve(label.size() * 2);
    for (char c : label) {
        switch (c) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '<':
                result += "\\<";
                break;
            case '>':
                result += "\\>";
                break;
            case '{':
                result += "\\{";
                break;
            case '}':
                result += "\\}";
                break;
            case '\n':
                result += "\\n";
                break;
            default:
                result += c;
                break;
        }
    }
    return result;
}