# onnx-graph

ONNX tensor compiler — loads a neural network from an `.onnx` file,
builds a typed computational graph, and compiles it through MLIR to LLVM IR and native assembly.

## What it does

- Parses binary `.onnx` files via protobuf
- Builds a typed computational graph (DAG): nodes = ops, tensors = edges
- Supports ops: **Conv, Relu, Add, Mul, MatMul, Gemm** with all their attributes
- Topological sort for correct execution order (cycle detection included)
- Graphviz visualization (`.dot` → `.png`)
- MLIR codegen using the `linalg` dialect
- Full lowering pipeline: MLIR → LLVM IR → assembly (x86-64, aarch64, ...)

## Compilation pipeline

```text
  model.onnx
      │  onnx_parser
      ▼
  Graph (DAG)
      │  MLIRCodegen
      ▼
  model.mlir          (linalg + arith + memref dialects)
      │  mlir-opt  (linalg→loops → scf → cf → arith→llvm → func→llvm)
      ▼
  model_lowered.mlir
      │  mlir-translate --mlir-to-llvmir
      ▼
  model.ll            (LLVM IR)
      │  llc
      ▼
  model.s             (assembly)
```

## Graph structure example

```text
  input
    │
  [Conv]
    │
  conv_out
    │
  [Relu]
    │
  relu_out
    │
  [Gemm]
    │
  output
```

## Dependencies

```bash
# Core
sudo apt install cmake build-essential protobuf-compiler libprotobuf-dev graphviz

# MLIR/LLVM codegen (Ubuntu 22.04+)
sudo apt install mlir-18-tools llvm-18

# Comparison script
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime
```

## Build

```bash
# 1. Download ONNX protobuf schema
mkdir -p third_party/onnx
wget -O third_party/onnx/onnx.proto3 \
  https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3
mv third_party/onnx/onnx.proto3 third_party/onnx/onnx.proto

# 2. Configure and build
mkdir build && cd build
protoc --cpp_out=. -I ../third_party/onnx ../third_party/onnx/onnx.proto
cmake ..
make -j$(nproc)
```

## Run

```bash
cd build

# Manual graph demo (no .onnx file needed)
./onnx_graph

# Load and inspect an ONNX model
./onnx_graph model.onnx

# Emit MLIR
./onnx_graph --emit-mlir --output out model.onnx
# → out.mlir

# Emit LLVM IR
./onnx_graph --emit-llvmir --output out model.onnx
# → out.mlir  out_lowered.mlir  out.ll

# Emit assembly (x86-64 by default)
./onnx_graph --emit-asm --output out model.onnx
# → out.mlir  out_lowered.mlir  out.ll  out.s

# Different target and optimization level
./onnx_graph --emit-asm --target aarch64 --opt-level 3 --output out model.onnx

# Custom tool paths
./onnx_graph --emit-asm \
  --mlir-opt mlir-opt \
  --mlir-translate mlir-translate \
  --llc llc \
  --output out model.onnx

# Visualize graph
./onnx_graph --visualize graph.png model.onnx
```

All CLI flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--emit-mlir` | off | Generate `<output>.mlir` |
| `--emit-llvmir` | off | Also generate `<output>.ll` |
| `--emit-asm` | off | Also generate `<output>.s` |
| `--output <name>` | `model` | Output file prefix |
| `--target <arch>` | `x86-64` | Target architecture for llc |
| `--opt-level <n>` | `2` | Optimization level (0–3) |
| `--mlir-opt <path>` | `mlir-opt-18` | Path to mlir-opt |
| `--mlir-translate <path>` | `mlir-translate-18` | Path to mlir-translate |
| `--llc <path>` | `llc-18` | Path to llc |
| `--visualize <out.png>` | — | Render graph to PNG |

## Comparison with onnxruntime

The script generates small ONNX models (Add, Mul, Relu, MatMul, Gemm) via PyTorch,
runs them with onnxruntime as reference, then runs our compiler pipeline
and shows the generated MLIR.

```bash
# from repo root
python3 scripts/compare_with_pytorch.py

# single op
python3 scripts/compare_with_pytorch.py --test Relu

# custom build dir or tool paths
python3 scripts/compare_with_pytorch.py --build-dir build --mlir-opt mlir-opt-18
```

## Tests

```bash
cd build

# Run all tests
./tests/tests

# Run a specific suite
./tests/tests --gtest_filter=MLIRCodegenTest.*
./tests/tests --gtest_filter=GraphTest.*

# Run with visualization output
cd ..
chmod +x run_tests_with_viz.sh
./run_tests_with_viz.sh
# Output .dot / .png files go to test_viz_output/
```

## Project structure

```text
onnx-graph/
├── include/
│   ├── tensor.hpp           # Tensor, TensorShape, DataType
│   ├── node.hpp             # Node, AttrValue
│   ├── graph.hpp            # Graph (DAG)
│   ├── graph_utils.hpp      # MakeNode, MakeTensor helpers
│   ├── onnx_loader.hpp      # binary .onnx file I/O
│   ├── onnx_parser.hpp      # protobuf → Graph
│   ├── visualizer.hpp       # Graphviz .dot generator
│   ├── mlir_codegen.hpp     # Graph → MLIR (linalg dialect)
│   └── compiler_driver.hpp  # MLIR → LLVM IR → assembly pipeline
├── src/
│   ├── graph.cpp
│   ├── onnx_loader.cpp
│   ├── onnx_parser.cpp
│   ├── visualizer.cpp
│   ├── mlir_codegen.cpp
│   ├── compiler_driver.cpp
│   └── main.cpp
├── tests/
│   ├── tests.cpp            # Google Test suite (graph + MLIRCodegen tests)
│   └── CMakeLists.txt
├── scripts/
│   └── compare_with_pytorch.py  # onnxruntime reference vs our compiler
├── third_party/
│   └── onnx/
│       └── onnx.proto
├── CMakeLists.txt
├── .clang-format
└── run_tests_with_viz.sh
```
