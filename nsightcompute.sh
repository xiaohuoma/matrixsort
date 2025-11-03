export PATH="/usr/local/cuda/nsight-compute-2023.2.0/target/linux-x64:$PATH"
export NVCOMPUTE="$HOME/.nsight-compute"
#   smsp__cycles_active.avg: 每个流多处理器（SM）的平均活跃时钟周期数 
#   smsp__inst_executed.avg.per_cycle_active: 每个活跃周期内平均执行的指令数（IPC）
# sudo -E env "PATH=$PATH" "NVCOMPUTE=$NVCOMPUTE" ncu --metrics smsp__cycles_active.avg,smsp__inst_executed.avg.per_cycle_active ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 8 > test.txt
# sudo -E env "PATH=$PATH" "NVCOMPUTE=$NVCOMPUTE" ncu --metrics smsp__sass_thread_inst_executed_op_global_ld.sum,smsp__sass_thread_inst_executed_op_shared_ld.sum ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 8 > test.txt

#   --set full: 启用详细指标收集
#   -o report: 输出文件名（生成 .ncu-rep 文件）
# sudo -E env "PATH=$PATH" "NVCOMPUTE=$NVCOMPUTE" ncu --set full \
#     ./hunyuangraph /media/jiangdie/新加卷/graph_10w/audikw_1.graph 8 > test.txt

#   使用 --kernel-name 替代 --kernel-regex
sudo -E env "PATH=$PATH" "NVCOMPUTE=$NVCOMPUTE" ncu --set full --kernel-name "set_gain_val_warp" \
  ./hunyuangraph /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 8 1 >> set_gain_val_warp.txt