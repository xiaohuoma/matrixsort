# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/1138_bus.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/bcsstk08.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/can_1054.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/bp_800.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/rdb450l.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/oscil_dcop_11.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/fs_183_6.graph 8 >> test.txt
# ./mygpmetis /media/jiangdie/新加卷/graph_less10w/jgl011.graph 8 >> test.txt

input="graph1.csv"

i=0 # 初始化计数器
while IFS=',' read -r Name; do
    # ./myndmetis "/media/jiangdie/shm_ssh/graph_10w/${Name}.graph" >> test.txt
    # ./myndmetis /media/jiangdie/shm_ssd/graph_10w/$Name.graph >> test.txt
    # ./mygpmetis /media/jiangdie/shm_ssd/graph_less10w/$Name.graph 8 >> test.txt
    ./mygpmetis graph/${Name}_20.graph 2 >> test.txt
    # python3 log_analyze.py ${Name}_log.txt
    i=$((i + 1)) # 更新计数器
done < "$input"

echo "Processed $i files."