#!/bin/bash

current_path=$(pwd)
cd matrix_to_graph || { echo "Failed to enter matrix_to_graph/"; exit 1; }
gcc matrix_to_graph.c -o matrix_to_graph -O3
cd .. || exit 1

current_path=$(pwd)
cd graph_to_grf || { echo "Failed to enter graph_to_grf/"; exit 1; }
gcc graph_to_grf.c -o graph_to_grf -O3
cd .. || exit 1

input="matrix.csv"

mkdir -p matrices
mkdir -p graphs
cd matrices || exit 1

PREFIX="http://sparse-files.engr.tamu.edu"

i=1
while IFS=',' read -r Name; do
    # 处理文件名
    filename=$(basename "$Name" .tar.gz)

    start_time=$(date +%s)

    # 下载文件
    if ! wget "$PREFIX${Name}"; then
        echo "Failed to download ${Name}"
        continue
    fi

    # 解压
    # echo tar -xf "$filename".tar.gz
    tar -xf "$filename".tar.gz

    # 移动 .mtx 文件到 matrices/
    if [ -f "${filename}/${filename}.mtx" ]; then
        mv "${filename}/${filename}.mtx" .
    else
        echo "MTX file not found in ${filename}/"
        continue
    fi

    rm -rf "$filename".tar.gz
    rm -rf "$filename"

    # 处理矩阵
    "${current_path}/matrix_to_graph/matrix_to_graph" "${filename}.mtx"

    # 处理grf
    "${current_path}/graph_to_grf/graph_to_grf" "${filename}.graph" > "${current_path}/graphs/${filename}.grf"

    mv "${filename}.graph" ${current_path}/graphs

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Processed $i files: ${Name} | Duration: ${duration} seconds"
    i=$((i + 1))
done < "${current_path}/${input}"