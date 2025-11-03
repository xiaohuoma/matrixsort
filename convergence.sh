#!/bin/bash
# set -e  # 开启错误中断

i_values=(3 5 8 10 15 20 25 30)
j_values=(2 5 8 10)
k_values=(1 2 3 5 8 10)  # 统一小数格式

for i in "${i_values[@]}"; do
    for j in "${j_values[@]}"; do
        for k in "${k_values[@]}"; do
            # echo "Running: i=$i, j=$j, k=$k"
            # echo "$i"
            # echo "$j"
            # echo "$k"
            # python3 convergence.py "$i" \
            #     --delta-window "$j" \
            #     --delta-epsilon "$k" \
            #     --k-factor 76708152
            python3 convergence2.py "$i" \
                --delta-window "$j" \
                --delta-epsilon "$k" \
                --k-factor 76708152 \
                --k-minedgecut 234140
        done
    done
done