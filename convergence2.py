import argparse
from itertools import islice

def positive_int(value):
    """验证输入是否为正整数"""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} 必须为正整数")
    return ivalue

def non_negative_float(value):
    """验证输入是否为非负数"""
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} 必须为非负数")
    return fvalue

# 解析命令行参数
parser = argparse.ArgumentParser(description="检测最小值收敛性")
parser.add_argument(
    "threshold",
    type=positive_int,
    help="连续出现该次数的非更小值时判定收敛"
)
parser.add_argument(
    "--delta-window",
    type=positive_int,
    default=3,
    help="连续差值递减的次数要求（默认：3）"
)
parser.add_argument(
    "--delta-epsilon",
    type=non_negative_float,
    default=0.0,
    help="允许的差值阈值基准值（默认：0.0）"
)
parser.add_argument(
    "--k-factor",
    type=positive_int,
    default=10000,
    help="动态阈值系数，公式：(k * delta_epsilon)/10000（默认：10000）"
)
parser.add_argument(
    "--k-minedgecut",
    type=positive_int,
    default=10000,
    help="初始最小值设定（默认：10000）"
)
args = parser.parse_args()

# 计算动态阈值
dynamic_threshold = (args.k_factor * args.delta_epsilon) / 10000

current_min = args.k_minedgecut  # 直接使用参数初始化
previous_min = None
delta_history = []
counter = 0

with open('input2.txt', 'r') as file:
    # 跳过前120行，从第121行开始读取
    for line in islice(file, 120, None):
        num = int(line.strip())
        if num < current_min:
            if previous_min is not None:
                delta = previous_min - num
                delta_history.append(delta)
                
                # 严格保留delta_window的判断
                if len(delta_history) >= args.delta_window:
                    recent_deltas = delta_history[-args.delta_window:]
                    
                    # 条件1: 差值连续递减
                    is_decreasing = all(recent_deltas[i] > recent_deltas[i+1] 
                                      for i in range(len(recent_deltas)-1))
                    
                    # 条件2: 最后差值 <= 动态阈值
                    meets_threshold = recent_deltas[-1] <= dynamic_threshold
                    
                    if is_decreasing and meets_threshold:
                        # print(f"差值收敛于: {num}")
                        # print(f"条件: 连续{args.delta_window}次递减 | 最后差值({recent_deltas[-1]}) ≤ {dynamic_threshold}")
                        print(f"{current_min}")
                        exit()
            
            previous_min = current_min
            current_min = num
            counter = 0
        else:
            counter += 1
            if counter >= args.threshold:
                # print(f"连续未更新收敛: {current_min}（连续{args.threshold}次未更新）")
                print(f"{current_min}")
                exit()

# print(f"文件读取完毕，最终最小值: {current_min}（未触发任何收敛条件）")
print(f"{current_min}")