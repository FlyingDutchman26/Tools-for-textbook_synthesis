import json
import os
import random

folder_path = './textbook'
size = len(os.listdir(folder_path))
random_file = os.listdir(folder_path)[random.randint(0,size)]
with open('./textbook/'+random_file,'r') as f:
    res = json.load(f)
    print(res['response'])