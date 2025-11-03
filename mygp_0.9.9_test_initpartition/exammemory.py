import re

def process_file(file_path):
    ptr_operations = {}  # 用于存储每个指针值及其对应的malloc、free和realloc的字节数

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ptr='):
                ptr_value = line.split('ptr=')[1].split()[0]  # 提取指针值
                if ptr_value not in ptr_operations:
                    # ptr_operations[ptr_value] = {'malloc_bytes': 0, 'free_bytes': 0, 'realloc_bytes': 0}
                    ptr_operations[ptr_value] = {'nbytes': 0}

                # 尝试提取nbytes=后面的数字
                nbytes_match = re.search(r'nbytes=(\d+)', line)
                if nbytes_match:
                    nbytes = int(nbytes_match.group(1))
                else:
                    continue  # 如果没有找到nbytes=后面的数字，则跳过当前行

                # 根据行中是否包含malloc=、free=或realloc=来更新字节数
                if 'malloc=' in line:
                    ptr_operations[ptr_value]['nbytes'] += nbytes
                elif 'free=' in line:
                    ptr_operations[ptr_value]['nbytes'] -= nbytes
                    if ptr_operations[ptr_value]['nbytes'] != 0:
                        print(f"Pointer {ptr_value} {line}")
                elif 'realloc=' in line:
                    ptr_operations[ptr_value]['nbytes'] = nbytes  # 直接赋值为nbytes

    return ptr_operations

def print_operations(ptr_operations):
    # 输出每个指针字符串后面malloc=、free=和realloc=的字节数
    # for ptr, operations in ptr_operations.items():
    #     print(f"Pointer {ptr}:")
    #     print(f"  nbytes: {operations['nbytes']}")
        # print(f"  Malloc bytes: {operations['malloc_bytes']}")
        # print(f"  Free bytes: {operations['free_bytes']}")
        # print(f"  Realloc bytes: {operations['realloc_bytes']}")

    # 找出malloc、free和realloc字节数不对等的指针值
    for ptr, operations in ptr_operations.items():
        if operations['nbytes'] != 0:
            print(f"Pointer {ptr} has unequal malloc and free bytes or non-zero realloc bytes: nbytes={operations['nbytes']}")

# 替换'your_file.txt'为你的文件路径
file_path = 'test.txt'
ptr_operations = process_file(file_path)
print_operations(ptr_operations)