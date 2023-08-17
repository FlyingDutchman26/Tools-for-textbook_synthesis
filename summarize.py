import json
import os

import tiktoken

# 指定包含文件的文件夹路径
folder_path = './textbook'


total_tokens = 0

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.json'):
#         # 打开文件
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             response = data.get('response', '')  # 根据实际JSON结构修改键名
#             # 使用tokenizer计算token数量
#             tokens = num_tokens_from_string(response,'gpt2')
#             total_tokens += tokens

print(f'Total response:{len(os.listdir(folder_path))}')
print(f"Total tokens in the folder: {total_tokens}")
