import re

# def check_memory_leaks(log_content):
#     # 使用字典来记录每个内存分配的指针和大小
#     memory_allocations = {
#         'lmalloc': [],
#         'rmalloc': [],
#     }
    
#     # 解析日志内容
#     for line in log_content.splitlines():
#         # 匹配lmalloc和rmalloc的行
#         lmalloc_match = re.search(r'lmalloc lmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
#         rmalloc_match = re.search(r'rmalloc rmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
#         lfree_match = re.search(r'lfree\s+ lmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
#         rfree_match = re.search(r'rfree\s+ rmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
        
#         if lmalloc_match:
#             used_size = int(lmalloc_match.group(1))
#             memory_allocations['lmalloc'].append(used_size)
        
#         if rmalloc_match:
#             used_size = int(rmalloc_match.group(1))
#             memory_allocations['rmalloc'].append(used_size)
        
#         if lfree_match:
#             used_size = int(lfree_match.group(1))
#             if memory_allocations['lmalloc']:
#                 last_size = memory_allocations['lmalloc'].pop()
#                 if last_size != used_size:
#                     # message = re.search(r'lfree\s+ lmove_pointer=.*? size=\s*\d+ used_size=\s*\d+ (\S+)', line)
#                     message = re.search(r'lfree\s+lmove_pointer=.*? size=\s*\d+ used_size=\s*\d+\s+(.*)', line)
#                     print(f"Left free size mismatch: expected {last_size} got {used_size} message {message.group(1)}")
        
#         if rfree_match:
#             used_size = int(rfree_match.group(1))
#             if memory_allocations['rmalloc']:
#                 last_size = memory_allocations['rmalloc'].pop()
#                 if last_size != used_size:
#                     # message = re.search(r'rfree\s+ rmove_pointer=.*? size=\s*\d+ used_size=\s*\d+ (\S+)', line)
#                     message = re.search(r'rfree\s+ rmove_pointer=.*? size=\s*\d+ used_size=\s*\d+\s+(.*)', line)
#                     print(f"Right free size mismatch: expected {last_size} got {used_size} message {message.group(1)}")
#     # 检查未释放的内存
#     leaks = {
#         'lmalloc_leaks': memory_allocations['lmalloc'],
#         'rmalloc_leaks': memory_allocations['rmalloc'],
#     }

#     return leaks

def check_memory_leaks(log_content):
    memory_allocations = {
        'lmalloc': [],
        'rmalloc': [],
    }
    
    # 新增状态控制标志（网页1）
    check_lmalloc = True  # 初始允许检查

    for line in log_content.splitlines():
        # 新增行为检测逻辑（网页1的循环检测思路）
        if 'record_lmove_pointer' in line:
            check_lmalloc = False
            continue
        elif 'return_lmove_pointer' in line:
            check_lmalloc = True
            continue

        # 根据状态决定是否检查lmalloc相关操作
        if check_lmalloc:
            lmalloc_match = re.search(r'lmalloc lmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
            lfree_match = re.search(r'lfree\s+ lmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
            
            if lmalloc_match:
                used_size = int(lmalloc_match.group(1))
                memory_allocations['lmalloc'].append(used_size)
            
            if lfree_match:
                used_size = int(lfree_match.group(1))
                if memory_allocations['lmalloc']:
                    last_size = memory_allocations['lmalloc'].pop()
                    if last_size != used_size:
                        message = re.search(r'lfree\s+lmove_pointer=.*? size=\s*\d+ used_size=\s*\d+\s+(.*)', line)
                        print(f"Left free size mismatch: expected {last_size} got {used_size} message {message.group(1)}")

        # rmalloc检查始终保持启用
        rmalloc_match = re.search(r'rmalloc rmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
        rfree_match = re.search(r'rfree\s+ rmove_pointer=.*? size=\s*\d+ used_size=\s*(\d+)', line)
        
        if rmalloc_match:
            used_size = int(rmalloc_match.group(1))
            memory_allocations['rmalloc'].append(used_size)
        
        if rfree_match:
            used_size = int(rfree_match.group(1))
            if memory_allocations['rmalloc']:
                last_size = memory_allocations['rmalloc'].pop()
                if last_size != used_size:
                    message = re.search(r'rfree\s+ rmove_pointer=.*? size=\s*\d+ used_size=\s*\d+\s+(.*)', line)
                    print(f"Right free size mismatch: expected {last_size} got {used_size} message {message.group(1)}")

    leaks = {
        'lmalloc_leaks': memory_allocations['lmalloc'],
        'rmalloc_leaks': memory_allocations['rmalloc'],
    }
    return leaks

# 读取内存日志文件
with open('memory_check.txt', 'r') as file:
    log_content = file.read()

# 检查内存泄漏
leaks = check_memory_leaks(log_content)

# 输出结果
if leaks['lmalloc_leaks'] or leaks['rmalloc_leaks']:
    print("Memory Leaks Detected:")
    if leaks['lmalloc_leaks']:
        print(f"  Left malloc leaks: {leaks['lmalloc_leaks']}")
    if leaks['rmalloc_leaks']:
        print(f"  Right malloc leaks: {leaks['rmalloc_leaks']}")
else:
    print("No memory leaks detected.")