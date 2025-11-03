import re

def extract_gpu_edgecut(file_path):
    edgecuts_init = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            edgecut_match = re.search(r'best_edgecut=\s*([\d.]+)', line)
            if edgecut_match:
                edgecut_value = int(edgecut_match.group(1))
                edgecuts_init.append(edgecut_value)  # 直接使用v的值作为索引
    return edgecuts_init

def extract_graph_names(file_path):
    graph_names = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 匹配路径后的内容，直到遇到 .graph 或空格
            path_match = re.search(r'graph:/media/jiangdie/新加卷/graph_10w/(.*?)(?=\.graph|\s|$)', line)
            if path_match:
                extracted_str = path_match.group(1)
                graph_names.append(extracted_str)
    return graph_names

def extract_cpu_edgecut(file_path):
    edgecuts_init = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'cpu\s*([\s.]+)', line)
            idx_match = re.search(r'v=\s*([\d.]+)', line)
            edgecut_match = re.search(r'edgecut=\s*([\d.]+)', line)
            if match and idx_match and edgecut_match:
                idx = int(idx_match.group(1))  # 假设v的值是整数
                edgecut_value = int(edgecut_match.group(1))
                edgecuts_init[idx] = edgecut_value  # 直接使用v的值作为索引
    return edgecuts_init

def extract_gpu_nvtxs(file_path):
    nvtxs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'cnvtxs=\s*([\d.]+)', line)
            if match:
                nvtx_value = int(match.group(1))
                nvtxs.append(nvtx_value)
    return nvtxs

def extract_gpu_nedges(file_path):
    nedges = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'cnedges=\s*([\d.]+)', line)
            if match:
                nedge_value = int(match.group(1))
                nedges.append(nedge_value)
    return nedges

def extract_gpu_inittimes(file_path):
    inittimes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'gpu_Bisection_time\s*([\d.]+)', line)
            if match:
                inittime_value = float(match.group(1))
                inittimes.append(inittime_value)
    return inittimes

def extract_cpu_BFStimes(file_path):
    BFStimes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'random_match_time\s*([\d.]+)', line)
            if match:
                BFStime_value = float(match.group(1))
                BFStimes.append(BFStime_value)
    return BFStimes

def extract_cpu_computetimes(file_path):
    computetimes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'Compute Partition Inf 2way:\s*([\d.]+)', line)
            if match:
                computetime_value = float(match.group(1))
                computetimes.append(computetime_value)
    return computetimes

file_path = 'test.txt'
graph_names = extract_graph_names(file_path)
# gpu_edgecuts = extract_gpu_edgecut(file_path)
# cpu_edgecuts = extract_cpu_edgecut(file_path)
# gpu_nvtxs = extract_gpu_nvtxs(file_path)
# gpu_nedges = extract_gpu_nedges(file_path)
# gpu_inittimes = extract_gpu_inittimes(file_path)
# cpu_BFStimes = extract_cpu_BFStimes(file_path)
# cpu_computetimes = extract_cpu_computetimes(file_path)

# sorted(gpu_edgecuts.items())
# sorted(cpu_edgecuts.items())
# print(len(gpu_edgecuts), len(cpu_edgecuts))
# if len(gpu_edgecuts) == 0:
#     print("gpu_edgecuts values not found in the file.")
# for ptr in range(len(gpu_edgecuts)):
#     # print(ptr, gpu_edgecuts[ptr], cpu_edgecuts[ptr], gpu_edgecuts[ptr] == cpu_edgecuts[ptr])
#     if gpu_edgecuts[ptr] != cpu_edgecuts[ptr]:
#         print(ptr, gpu_edgecuts[ptr], cpu_edgecuts[ptr], gpu_edgecuts[ptr] == cpu_edgecuts[ptr])
#     # else:
#     #     print(ptr)

# 打印edgecuts，索引从0开始
if graph_names:
    for graph_name in graph_names:
        print(graph_name)
else:
    print("graph_names values not found in the file.")

# # 打印edgecuts，索引从0开始
# if gpu_edgecuts:
#     for edgecut in gpu_edgecuts:
#         print(edgecut)
# else:
#     print("gpu_edgecuts values not found in the file.")

# # 打印edgecuts，索引从0开始
# if cpu_edgecuts:
#     for idx, edgecut in sorted(cpu_edgecuts.items()):
#         print(f"v={idx}: edgecut={edgecut}")
# else:
#     print("cpu_edgecuts values not found in the file.")

# # nvtxs
# if gpu_nvtxs:
#     for nvtx in gpu_nvtxs:
#         print(nvtx)
# else:
#     print("gpu_nvtxs values not found in the file.")

# # nedges
# if gpu_nedges:
#     for nedge in gpu_nedges:
#         print(nedge)
# else:
#     print("gpu_nedges values not found in the file.")

# inittimes
# if gpu_inittimes:
#     ave = 0
#     for inittime in gpu_inittimes:
#         print(inittime)
#         ave += inittime
#     print(ave/len(gpu_inittimes))
# else:
#     print("gpu_inittimes values not found in the file.")

# # BFStimes
# if cpu_BFStimes:
#     all_BFStime = 0
#     for BFStime in cpu_BFStimes:
#         all_BFStime += BFStime
#     print(all_BFStime)
# else:
#     print("cpu_BFStimes values not found in the file.")

# # BFStimes
# if cpu_BFStimes:
#     for BFStime in cpu_BFStimes:
#         print(BFStime)
# else:
#     print("cpu_BFStimes values not found in the file.")

# # computetimes
# if cpu_computetimes:
#     all_computetime = 0
#     for computetime in cpu_computetimes:
#         all_computetime += computetime
#     print(all_computetime)
# else:
#     print("cpu_computetimes values not found in the file.")