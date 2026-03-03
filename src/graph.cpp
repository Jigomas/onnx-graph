#include "../include/graph.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

Graph::Graph(const std::string& name) : name(name) {}

Graph Graph::Clone() const {
    Graph copy(name);

    for (const auto& node_ptr : nodes) {
        copy.nodes.push_back(std::make_unique<Node>(*node_ptr));
    }

    copy.tensors = tensors;
    copy.inputs = inputs;
    copy.outputs = outputs;

    return copy;
}

void Graph::AddNode(std::unique_ptr<Node> node) {
    if (!node)
        throw std::invalid_argument("Graph::AddNode: node must not be null");

    assert(!node->op_type.empty() && "AddNode: node op_type must not be empty");

    if (FindNode(node->name) != nullptr)
        throw std::invalid_argument("Graph::AddNode: node with name '" + node->name +
                                    "' already exists");

    nodes.push_back(std::move(node));
}

Node* Graph::FindNode(const std::string& name) const {
    for (const auto& node_ptr : nodes) {
        if (node_ptr->name == name) {
            return node_ptr.get();
        }
    }

    return nullptr;
}

std::optional<Tensor> Graph::FindTensor(const std::string& target_name) const {
    auto it = tensors.find(target_name);
    if (it == tensors.end())
        return std::nullopt;

    return it->second;
}

void Graph::DumpGraph() const {
    std::cout << " Graph:   " << name << '\n';
    std::cout << " Nodes:   " << nodes.size() << '\n';
    std::cout << " Tensors: " << tensors.size() << '\n';

    std::cout << " Inputs:  ";
    for (const auto& inp : inputs)
        std::cout << inp << ' ';
    std::cout << '\n';

    std::cout << " Outputs: ";
    for (const auto& out : outputs)
        std::cout << out << ' ';
    std::cout << '\n';

    for (const auto& node_ptr : nodes) {
        std::cout << " [" << node_ptr->op_type << "] " << node_ptr->name << '\n';
        for (const auto& [key, val] : node_ptr->attributes) {
            std::cout << "   attr: " << key << '\n';
        }
    }
}

std::vector<Node*> Graph::TopologicalSort() const {
    std::unordered_map<std::string, Node*> tensor_producer;
    for (const auto& node_ptr : nodes) {
        for (const auto& output_name : node_ptr->outputs) {
            tensor_producer[output_name] = node_ptr.get();
        }
    }

    std::unordered_map<std::string, VisitState> visited;
    std::vector<Node*> result;

    result.reserve(nodes.size());

    for (const auto& node_ptr : nodes) {
        if (visited[node_ptr->name] == VisitState::WHITE) {
            TopologicalSortData(node_ptr.get(), visited, tensor_producer, result);
        }
    }

    return result;
}

void Graph::TopologicalSortData(Node* node,
                                std::unordered_map<std::string, VisitState>& visited,
                                const std::unordered_map<std::string, Node*>& tensor_producer,
                                std::vector<Node*>& result) const {
    visited[node->name] = VisitState::GRAY;

    for (const auto& input_name : node->inputs) {
        auto it = tensor_producer.find(input_name);
        if (it == tensor_producer.end())
            continue;

        Node* producer = it->second;

        if (visited[producer->name] == VisitState::GRAY)
            throw std::runtime_error("Cycle detected at node: " + producer->name);

        if (visited[producer->name] == VisitState::WHITE)
            TopologicalSortData(producer, visited, tensor_producer, result);
    }

    visited[node->name] = VisitState::BLACK;
    result.push_back(node);
}