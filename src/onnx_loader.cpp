#include "../include/onnx_loader.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../include/onnx_parser.hpp"
#include "onnx.pb.h"

Graph OnnxLoader::Load(const std::string& filepath) {
    if (filepath.empty())
        throw std::invalid_argument("OnnxLoader::Load: filepath is empty");

    if (!std::filesystem::exists(filepath))
        throw std::runtime_error("OnnxLoader::Load: file not found: " + filepath);

    if (std::filesystem::file_size(filepath) == 0)
        throw std::runtime_error("OnnxLoader::Load: file is empty: " + filepath);

    std::ifstream file(filepath, std::ios::binary);
    if (!file)
        throw std::runtime_error("OnnxLoader::Load: cannot open: " + filepath);

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&file))
        throw std::runtime_error("OnnxLoader::Load: failed to parse ONNX: " + filepath);

    return OnnxParser().Parse(model);
}