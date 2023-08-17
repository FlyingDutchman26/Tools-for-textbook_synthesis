import re

# 读取文件内容
with open('apis.txt', 'r') as file:
    lines = file.readlines()

# 提取每一行的key
extracted_keys = []
for line in lines:
    match = re.search(r'sk-[^\|]+', line)
    if match:
        extracted_keys.append(match.group(0))

# 重新储存到新文件
with open('extracted_keys.txt', 'w') as file:
    file.writelines('\n'.join(extracted_keys))
