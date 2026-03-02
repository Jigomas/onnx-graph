#!/bin/bash

mkdir -p test_viz_output
rm -rf test_viz_output/*

cd build

make clean
make -j4

./tests/tests --gtest_filter=*.* > test_output.log

for dot_file in *.dot; do
    if [ -f "$dot_file" ]; then
        base_name=$(basename "$dot_file" .dot)
        cp "$dot_file" "../test_viz_output/"
        dot -Tpng "$dot_file" -o "../test_viz_output/${base_name}.png"
        echo "Created test_viz_output/${base_name}.png"
        rm "$dot_file"
    fi
done

cd ..

echo ""
echo "All visualizations saved in test_viz_output/"
ls -la test_viz_output/