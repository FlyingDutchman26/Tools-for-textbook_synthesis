# 此脚本用于更新可用keys

# 读取failed_keys.txt中的内容
with open('failed_api_keys.txt', 'r') as failed_file:
    failed_keys = set(failed_file.read().splitlines())

# 读取extracted_keys.txt中的内容
with open('extracted_keys.txt', 'r') as extracted_file:
    extracted_keys = extracted_file.read().splitlines()

# 从extracted_keys中删除failed_keys
filtered_keys = [key for key in extracted_keys if key not in failed_keys]

# 将过滤后的keys重新储存到新文件
with open('extracted_keys.txt', 'w') as filtered_file:
    filtered_file.write('\n'.join(filtered_keys))
