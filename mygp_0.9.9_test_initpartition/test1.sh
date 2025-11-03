# ./mygpmetis /media/jiangdie/新加卷/graph_10w/af_shell6.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/audikw_1.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/cage13.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/AS365.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/hugebubbles-00000.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/rgg_n_2_22_s0.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_10w/vas_stokes_2M.graph 8 >> test.txt

./mygpmetis graph/audikw_1_gpu.graph 2 0 >> test.txt
./mygpmetis graph/rgg_n_2_22_s0_gpu.graph 2 0 >> test.txt
./mygpmetis graph/hugebubbles-00000_gpu.graph 2 0 >> test.txt
./mygpmetis graph/vas_stokes_2M_gpu.graph 2 0 >> test.txt

./mygpmetis graph/audikw_1_gpu.graph 8 0 >> test.txt
./mygpmetis graph/rgg_n_2_22_s0_gpu.graph 8 0 >> test.txt
./mygpmetis graph/hugebubbles-00000_gpu.graph 8 0 >> test.txt
./mygpmetis graph/vas_stokes_2M_gpu.graph 8 0 >> test.txt

./mygpmetis graph/audikw_1_gpu.graph 64 0 >> test.txt
./mygpmetis graph/rgg_n_2_22_s0_gpu.graph 64 0 >> test.txt
./mygpmetis graph/hugebubbles-00000_gpu.graph 64 0 >> test.txt
./mygpmetis graph/vas_stokes_2M_gpu.graph 64 0 >> test.txt

./mygpmetis graph/audikw_1_gpu.graph 256 0 >> test.txt
./mygpmetis graph/rgg_n_2_22_s0_gpu.graph 256 0 >> test.txt
./mygpmetis graph/hugebubbles-00000_gpu.graph 256 0 >> test.txt
./mygpmetis graph/vas_stokes_2M_gpu.graph 256 0 >> test.txt