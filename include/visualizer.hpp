#pragma once

#include "graph.hpp"

#include <string>



class Visualizer {
public:
    Visualizer()                                 = default;
    ~Visualizer()                                = default;
    Visualizer(const Visualizer&)                = default;
    Visualizer& operator=(const Visualizer&)     = default;
    Visualizer(Visualizer&&) noexcept            = default;
    Visualizer& operator=(Visualizer&&) noexcept = default;

    void ToDot(const Graph& graph, const std::string& filepath) const;

    void Render(const std::string& dot_file, const std::string& out_png) const;

private:
    std::string GetNodeColor(const std::string& op_type) const;
    std::string EscapeLabel(const std::string& label)    const;
};