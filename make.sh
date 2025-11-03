#!/bin/bash

# 第一步：查找nvcc编译器
NVCC_PATH=$(which nvcc 2>/dev/null)
if [ -z "$NVCC_PATH" ]; then
    echo "错误：未找到nvcc编译器，请安装CUDA Toolkit"
    exit 1
fi

# 第二步：生成临时CUDA文件（新增SM数量查询）
TMP_CU=$(mktemp /tmp/gpu_info.XXXXXX.cu)
cat << 'EOF' > $TMP_CU
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) return 1;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%d.%d,%d", prop.major, prop.minor, prop.multiProcessorCount);
    return 0;
}
EOF

# 第三步：编译并执行（输出格式改为"计算能力,SM数量"）
TMP_BIN="${TMP_CU%.cu}"
if ! nvcc $TMP_CU -o $TMP_BIN >/dev/null 2>&1; then
    echo "编译失败，请检查CUDA环境" >&2
    rm -f $TMP_CU $TMP_BIN
    exit 2
fi

OUTPUT=$($TMP_BIN 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "执行失败，可能缺少GPU或驱动异常" >&2
    rm -f $TMP_CU $TMP_BIN
    exit 3
fi

# 提取计算能力和SM数量
COMPUTE_CAP=$(echo $OUTPUT | cut -d, -f1)
SM_NUM=$(echo $OUTPUT | cut -d, -f2)

# 清理临时文件
rm -f $TMP_CU $TMP_BIN

# 输出结果
echo "GPU计算能力：$COMPUTE_CAP"
echo "SM数量：$SM_NUM"

# 第四部分：修改头文件（新增功能）
TARGET_FILE="hunyuangraph_define.h"
if [ ! -f "$TARGET_FILE" ]; then
    echo "错误：目标文件 $TARGET_FILE 不存在" >&2
    exit 4
fi

# 使用精确行号替换（网页6/7/8方法）
sed -i "7s/#define SM_NUM .*/#define SM_NUM $SM_NUM/" "$TARGET_FILE"

# 提取主次版本号（假设COMPUTE_CAP=8.6）
MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
MINOR=$(echo $COMPUTE_CAP | cut -d. -f2)

# 生成新的编译参数
NEW_ARCH="arch=compute_${MAJOR}${MINOR},code=sm_${MAJOR}${MINOR}"

# 替换Makefile中的错误参数
sed -i "s/-gencode\ arch=compute_[0-9]\+,\ code=sm_[0-9]\+/-gencode ${NEW_ARCH}/" Makefile

echo "已更新编译参数为：-gencode ${NEW_ARCH}"

current_path=$(pwd)
echo "current_path:${current_path}."

# figure 8 11 12 13 15
nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w

# figure 8 11 12
echo "Processing Hunyuangraph for figure 8 11 12."
input="graph_9.csv"
p_values="8 32 128 512"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1 >> 5090_hunyuan_1_${p}_graph9_time.txt
        
        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"

    mv "5090_hunyuan_1_${p}_graph9_time.txt" ${current_path}/data/hunyuan

    echo "Processed $p partitions."
done

# figure 13 15
echo "Processing Hunyuangraph for figure 13 15."
input="graph_all.csv"
p_values="8"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1 >> 5090_hunyuan_1_${p}_graphall_time.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"

    mv "5090_hunyuan_1_${p}_graphall_time.txt" ${current_path}/data/hunyuan

    echo "Processed $p partitions."
done

# figure 9
echo "Processing Hunyuangraph for figure 9."
nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE9_SUM
input="graph_9.csv"
p_values="8"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1 >> 5090_hunyuan_1_${p}_coarsen_adjwgtsum.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    mv "5090_hunyuan_1_${p}_coarsen_adjwgtsum.txt" ${current_path}/Figure/Figure9
    echo "Processed $p partitions."
done

nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE9_TIME
input="graph_9.csv"
p_values="8"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1 >> 5090_hunyuan_1_${p}_coarsen_time.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    mv "5090_hunyuan_1_${p}_coarsen_time.txt" ${current_path}/Figure/Figure9
    echo "Processed $p partitions."
done

# figure 10
echo "Processing Hunyuangraph for figure 10."
mkdir -p init_graphs
nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE10_CGRAPH

input="graph_9.csv"
p_values="1024"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1

        mv graph.txt ${Name}_gpu_1024.graph

        mv ${Name}_gpu_1024.graph ${current_path}/init_graphs
        
        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    echo "Processed $p partitions."
done

nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE10_EXHAUSTIVE

input="graph_9.csv"
p_values="2"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/init_graphs/${Name}_gpu_1024.graph $p 1 >> 5090_exhaustive_1024_${p}_1.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    
    mv 5090_exhaustive_1024_${p}_1.txt ${current_path}/Figure/Figure10/init_partition/exhaustive_gpu/

    echo "Processed $p partitions."
done

nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE10_SAMPLING

input="graph_9.csv"
p_values="2"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/init_graphs/${Name}_gpu_1024.graph $p 1 >> 5090_sampling_1024_${p}_1.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    
    mv 5090_sampling_1024_${p}_1.txt ${current_path}/Figure/Figure10/init_partition/sampling_hunyuan/

    echo "Processed $p partitions."
done

cd mygp_0.9.9_test_initpartition || { echo "Failed to enter mygp_0.9.9_test_initpartition/"; exit 1; }
gcc -O3 mygpmetis.c -o mygpmetis -lm
cd .. || exit 1

input="graph_9.csv"
p_values="2"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        "${current_path}/mygp_0.9.9_test_initpartition/mygpmetis" ${current_path}/init_graphs/${Name}_gpu_1024.graph 2 >> 5090_metis_1024_${p}_0_metis.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"
    
    mv 5090_metis_1024_${p}_0_metis.txt ${current_path}/Figure/Figure10/init_partition/metis_cpu/

    echo "Processed $p partitions."
done

# figure 14
echo "Processing Hunyuangraph for figure 14."
nvcc -std=c++11 -gencode ${NEW_ARCH} -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DFIGURE14_EDGECUT

input="graph_all.csv"
p_values="8"  # 改为字符串，用空格分隔

for p in $p_values; do  # 遍历空格分隔的值
    i=0
    while IFS=',' read -r Name; do
        ./hunyuangraph ${current_path}/graphs/${Name}.graph $p 1 >> 5090_hunyuan_1_${p}_graphall_edgecut.txt

        echo "Processed $i files: ${Name}.graph"
        i=$((i + 1))
    done < "$input"

    mv "5090_hunyuan_1_${p}_graphall_edgecut.txt" ${current_path}/data/hunyuan

    echo "Processed $p partitions."
done

# 需要检测的第三方包列表（兼容POSIX sh的写法）
required_packages="pandas numpy matplotlib"

# 获取当前Python命令（优先使用python3）
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "错误：未找到Python解释器"
    exit 1
fi

# 检测并安装缺失包
for pkg in $required_packages; do
    if ! $PYTHON_CMD -c "import $pkg" >/dev/null 2>&1; then
        echo "正在安装 $pkg..."
        if ! $PYTHON_CMD -m pip install --user $pkg; then
            echo "安装 $pkg 失败，请手动检查"
            exit 1
        fi
    else
        echo "$pkg 已安装"
    fi
done

echo "所有依赖已满足"

cd data
cd hunyuan
python3 hunyuan_9.py
python3 hunyuan_all.py
cd ..
cd ..

mkdir -p plot

cd Figure
cd Figure8
python3 figure8.py
mv figure8.pdf ${current_path}/plot
mv figure8.png ${current_path}/plot
cd ..

cd Figure9
python3 figure9.py
mv figure9.pdf ${current_path}/plot
mv figure9.png ${current_path}/plot
cd ..

cd Figure10
python3 figure10.py
mv figure10.pdf ${current_path}/plot
mv figure10.png ${current_path}/plot
cd ..

cd Figure11
python3 figure11.py
mv figure11.pdf ${current_path}/plot
mv figure11.png ${current_path}/plot
cd ..

cd Figure12
python3 figure12.py
mv figure12.pdf ${current_path}/plot
mv figure12.png ${current_path}/plot
cd ..

cd Figure13
python3 figure13.py
mv figure13.pdf ${current_path}/plot
mv figure13.png ${current_path}/plot
cd ..

cd Figure14
python3 figure14.py
mv figure14.pdf ${current_path}/plot
mv figure14.png ${current_path}/plot
cd ..

cd Figure15
python3 figure15.py
mv figure15.pdf ${current_path}/plot
mv figure15.png ${current_path}/plot
cd ..

cd ..
