import re
import sys
import matplotlib.pyplot as plt

# 时间
def extract_times(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'time=\s*([\d.]+)', line)
            if match:
                time_value = float(match.group(1))
                times.append(time_value)
    return times

# 类型
def extract_types(file_path):
    types = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'task_type=\s*([\d.]+)', line)
            if match:
                type_value = float(match.group(1))
                types.append(type_value)
    return types

# 大小
def extract_nbytes(file_path):
    nbytes = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'nbytes=\s*([\d.]+)', line)
            if match:
                nbyte_value = float(match.group(1))
                nbytes.append(nbyte_value)
    return nbytes

# 指针
def extract_ptrs(file_path):
    ptrs = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'ptr=(\S+?)\s', line)
            if match:
                ptr_value = match.group(1)
                ptrs.append(ptr_value)
    return ptrs

# 位置
def extract_locations(file_path):
    locations = []
    with open(file_path, 'r') as file:
        for line in file:
            # match = re.search(r'ptr=\s*([\d.]+)', line)
            match = re.search(r'located at (.*)', line)
            if match:
                location_value = match.group(1)
                locations.append(location_value)
    return locations

if len(sys.argv) < 2:
        print("Usage: python3 log_analyze.py <file_path>")
        sys.exit(1)

file_path = sys.argv[1]  # 获取命令行参数中的文件路径
# file_path = 'log.txt'
# times = extract_times(file_path)
# types = extract_types(file_path)
# nbytes = extract_nbytes(file_path)
# ptrs = extract_ptrs(file_path)
# locations = extract_locations(file_path)

times = []
types = []
nbytes = []
with open(file_path, 'r') as file:
    for line in file:
        # 将每行分割成三个部分
        parts = line.split()
        # 将分割后的字符串转换为浮点数或整数，并添加到相应的数组中
        times.append(float(parts[0]))
        types.append(int(parts[1]))
        nbytes.append(int(parts[2]))

# 颜色映射，可以根据需要自定义颜色
color_map = {1: 'red', 2: 'green', 3: 'blue'}
label_map = {1: 'malloc', 2: 'realloc', 3: 'free'}

# 使用数组索引作为x轴，nbytes的值作为y轴
# x_values = range(len(nbytes))
# 使用times数组的值作为x轴
x_values = times

# 创建折线图
plt.figure(figsize=(20, 5))  # 可以调整图形的大小

colors = ['black'] * len(types)  # 初始化颜色列表，默认颜色为 'black'
labels = [''] * len(types)  # 初始化标签列表，默认为空
# 为每种类型设置颜色和标签
for i in range(len(types)):
    colors[i] = color_map.get(types[i], 'black')
    labels[i] = label_map.get(types[i], '')

# # 绘制每个点，并根据types数组的值设置颜色
# 存储图例句柄
# legend_handles = {}
# for i in x_values:
#     line, = plt.plot(i, nbytes[i], marker='o', color=colors[i], label=labels[i], markersize=3)
#     # 只为每种类型的第一个点创建图例句柄
#     if labels[i] not in legend_handles:
#         legend_handles[labels[i]] = line
legend_handles = {}
for i in range(len(x_values)):
    line, = plt.plot(x_values[i], nbytes[i], marker='o', color=colors[i], label=labels[i], markersize=3)
    # 只为每种类型的第一个点创建图例句柄
    if labels[i] not in legend_handles:
        legend_handles[labels[i]] = line

# 添加标题和标签
plt.title('Memory Usage Over Time')
plt.xlabel('Index')
plt.ylabel('Memory Size (bytes)')

# 设置x轴起始值为0
# plt.xlim(0, len(nbytes))
# 设置x轴起始值和结束值
plt.xlim(0, max(x_values))

# 显示网格（可选）
plt.grid(True)

# 添加图例
legend_labels = ['malloc', 'realloc', 'free']
plt.legend(handles=legend_handles.values(), labels=legend_handles.keys(), ncol=3, loc='best', prop={'size': 20})

file_path = file_path.replace(".txt", "")
file_path += ".jpg"
# 显示图形
plt.savefig(file_path, format='jpg', bbox_inches='tight', dpi=300)

# 总时间
# if times:
#     for time in times:
#         print(time)
# else:
#     print("times values not found in the file.")

# # 类型
# if types:
#     for type_value in types:
#         print(type_value)
# else:
#     print("types values not found in the file.")

# # 大小
# if nbytes:
#     for nbyte in nbytes:
#         print(nbyte)
# else:
#     print("nbytes values not found in the file.")

# # 指针
# if ptrs:
#     for ptr in ptrs:
#         print(ptr)
# else:
#     print("ptrs values not found in the file.")

# # 位置
# if locations:
#     for location in locations:
#         print(location)
# else:
#     print("locations values not found in the file.")

# if times:
#     for time in times:
#         print(time)
# else:
#     print("times values not found in the file.")

# if types:
#     for type_value in types:
#         print(type_value)
# else:
#     print("types values not found in the file.")

# if nbytes:
#     for nbyte in nbytes:
#         print(nbyte)
# else:
#     print("nbytes values not found in the file.")