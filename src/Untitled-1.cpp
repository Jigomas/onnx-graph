#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../include/graph.hpp"

Graph::Graph(const std::string& name) : name(name) {}

Graph Graph::Clone() const {
    Graph copy(name)

        ;
