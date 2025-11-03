#!/bin/bash

# 循环从0到1138，执行mygpmetis命令
for i in $(seq 0 522)
do
    # ./mygpmetis graph/1138_bus_20.graph 2 $i
    ./mygpmetis graph/audikw_1_gpu.graph 2 $i >> test.txt
done