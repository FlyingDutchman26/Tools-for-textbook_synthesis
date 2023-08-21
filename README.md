<<<<<<< HEAD
# Tools for Code Synthesis using gpt-3.5-turbo
Fudan NLP

* Prepare an api lists for utilizing gpt-3.5-turbo(chatgpt)
* Use the scripts in this repo to generalize high-quality code datasets, which is a reproduction of **Textboos Are All You Need**

## code_synthesis_textbooks
* code_synthesis_textbooks.py 存储了生成时的知识领域(700多类) 以及 各种audience(受众) 参考TinyStories
* textbook_generation.py 用于并行调用gpt3.5-turbo api 生成textbooks数据集 
* summarize.py 用于统计生成的token数量，因统计较慢，估算一个文件约对应 1000 token
* demo.py 用于演示生成效果

## code_synthesis_exercises
* words.txt 储存了常用函数名称词汇以及leetcode中组成全部函数名称的词汇列表
* code_synthesis_exercises.py 用于读取 words.txt 的内容，构建练习的函数名称
* exercise_generation.py 用于并行调用gpt3.5-turbo api 生成exercise数据集 

## Tools
* openai_api_wrapper： 程博提供的调用openai的处理工具
* find_keys.py 从原生的openai_api列表中提取有用信息
* filter_keys 读取failed_api_keys中的失效key，并将其从extracted_keys.txt中过滤
