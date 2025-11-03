nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3 hunyuangraph.cu -o  hunyuangraph  --expt-relaxed-constexpr -w -DMEMORY_CHECK
./hunyuangraph /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 8 1 > memory_check.txt
python3 exammemory.py