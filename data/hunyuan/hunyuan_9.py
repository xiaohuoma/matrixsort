import pandas as pd
import re
import os
from pathlib import Path

def parse_log_file(file_path, base_dir):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    base_dir = str(base_dir).replace("/data/hunyuan", "")
    graph_dir = f"{base_dir}/graphs/"
    graph_sections = re.split(re.escape(graph_dir), content)[1:]
    
    graph_data = {}
    
    for section in graph_sections:
        graph_match = re.search(r'([\w-]+)\.graph', section)
        if not graph_match:
            continue
        graph_name = graph_match.group(1)
        
        runs = re.split(r'begin partition', section)[1:]
        
        for run in runs:
            run_data = {
                "Graph": graph_name,
                "Partition Time": None,
                "Coarsen Time": None,
                "Init Time": None,
                "Uncoarsen Time": None,
                "Edgecut": None,
                "Coarsen Edges": None,
                "Coarsen nvtxs": None
            }
            
            # 提取 Partition Time
            time_match = re.search(r'Hunyuangraph_Partition_time=\s+([\d.]+)\s+ms', run)
            if time_match:
                run_data["Partition Time"] = float(time_match.group(1))
            
            # 提取 Coarsen Time
            coarsen_match = re.search(r'------Coarsen_time=\s+([\d.]+)\s+ms', run)
            if coarsen_match:
                run_data["Coarsen Time"] = float(coarsen_match.group(1))
            
            # 提取 Init Time
            init_match = re.search(r'------Init_time=\s+([\d.]+)\s+ms', run)
            if init_match:
                run_data["Init Time"] = float(init_match.group(1))
            
            # 提取 Uncoarsen Time
            uncoarsen_match = re.search(r'------Uncoarsen_time=\s+([\d.]+)\s+ms', run)
            if uncoarsen_match:
                run_data["Uncoarsen Time"] = float(uncoarsen_match.group(1))
            
            # 提取 Edgecut
            edgecut_match = re.search(r'edge-cut=\s+(\d+)', run)
            if edgecut_match:
                run_data["Edgecut"] = int(edgecut_match.group(1))
            
            # 合并提取 Coarsen Edges 和 Coarsen nvtxs
            coarsen_end_match = re.search(r'Coarsen end: level=(\d+)\s+cnvtxs=(\d+)\s+cnedges=(\d+)', run)
            if coarsen_end_match:
                run_data["Coarsen Edges"] = int(coarsen_end_match.group(3))
                run_data["Coarsen nvtxs"] = int(coarsen_end_match.group(2))
            
            if run_data["Partition Time"] is not None:
                if graph_name not in graph_data:
                    graph_data[graph_name] = []
                graph_data[graph_name].append(run_data)
    
    best_runs = {}
    for graph_name, runs in graph_data.items():
        best_run = min(runs, key=lambda x: x["Partition Time"])
        best_runs[graph_name] = best_run  # 关键修复点
    
    return best_runs

# List of file suffixes to process (e.g., 8, 32, 128, 512)
suffixes = [8, 32, 128, 512]

# Base directory where the files are stored
# base_dir = r"D:\HuaweiMoveData\Users\huawei\Desktop\SC25\graph\data\hunyuan"
# 设置基础路径（相对路径的根目录）
base_dir = Path(__file__).parent.resolve()  # 获取当前脚本所在目录

# Loop through each suffix and generate corresponding CSV
for suffix in suffixes:
    file_name = f"5090_hunyuan_1_{suffix}_graph9_time.txt"
    file_path = os.path.join(base_dir, file_name)

    # Parse the log file
    best_runs = parse_log_file(file_path, base_dir)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(best_runs, orient='index')

    # Reorder columns for better readability
    column_order = [
        "Coarsen Edges",
        "Partition Time",
        "Coarsen Time",
        "Init Time",
        "Uncoarsen Time",
        "Edgecut",
        "Coarsen nvtxs"
    ]
    df = df[column_order]

    # Save to CSV
    output_csv_path = os.path.join(base_dir, f"5090_hunyuan_1_{suffix}_9.csv")
    df.to_csv(output_csv_path, index=True, index_label="Graph Name")

    print(f"CSV file saved to {output_csv_path}")
