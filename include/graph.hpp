#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "node.hpp"
#include "tensor.hpp"

enum class VisitState { WHITE, GRAY, BLACK };

class Graph {
public:
    explicit Graph(const std::string& name = "");
    ~Graph() = default;

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    Graph Clone() const;

    void AddNode(std::unique_ptr<Node> node);

    template <typename T, typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, Tensor>>>
    void AddTensor(T&& tensor) {
        if (tensor.name.empty())
            throw std::invalid_argument("Graph::AddTensor: tensor name must not be empty");

        const std::string key = tensor.name;

        if (tensors.count(key))
            throw std::invalid_argument("Graph::AddTensor: tensor '" + key + "' already exists");

        tensors.insert_or_assign(key, std::forward<T>(tensor));
    }

    Node* FindNode(const std::string& node_name) const;
    std::optional<Tensor> FindTensor(const std::string& tensor_name) const;

    std::vector<Node*> TopologicalSort() const;

    void DumpGraph() const;

    std::string name;
    std::vector<std::unique_ptr<Node>> nodes;
    std::unordered_map<std::string, Tensor> tensors;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

private:
    void TopologicalSortData(Node* node,
                             std::unordered_map<std::string, VisitState>& visited,
                             const std::unordered_map<std::string, Node*>& tensor_producer,
                             std::vector<Node*>& result) const;
};