# input="graph1.csv"

# {
# 	read
# 	i=1
# 	while IFS=',' read -r  Name 
# 	do

#         ./myndmetis /media/jiangdie/新加卷/graph_less10w/$Name.graph >> test.txt
	
# 		i=`expr $i + 1`
# 		done 
# } < "$input"
input="graph1.csv"

i=0 # 初始化计数器
while IFS=',' read -r Name; do
    # ./myndmetis "/media/jiangdie/shm_ssh/graph_10w/${Name}.graph" >> test.txt
    # ./myndmetis /media/jiangdie/shm_ssd/graph_10w/$Name.graph >> test.txt
    # ./myndmetis /media/jiangdie/新加卷/graph_10w/$Name.graph >> test.txt
    python3 log_analyze.py ${Name}_log.txt
    i=$((i + 1)) # 更新计数器
done < "$input"

echo "Processed $i files."