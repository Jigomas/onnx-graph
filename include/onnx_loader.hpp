#pragma once

#include <string>

#include "graph.hpp"

class OnnxLoader {
public:
    OnnxLoader() = default;
    ~OnnxLoader() = default;
    OnnxLoader(const OnnxLoader&) = default;
    OnnxLoader& operator=(const OnnxLoader&) = default;
    OnnxLoader(OnnxLoader&&) noexcept = default;
    OnnxLoader& operator=(OnnxLoader&&) noexcept = default;

    Graph Load(const std::string& path);
};