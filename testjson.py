import json

def read_jsonl_and_return_keys(file_path):
    keys_list = []  # 用于存储每个JSON对象的键

    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)  # 解析每一行为JSON对象
            keys = list(json_obj.keys())  # 获取JSON对象的键
            keys_list.append(keys)  # 将键添加到列表中

    return keys_list

file_path = r'D:\openAlexSnapshot\load_snapshot\works\updated_date=2023-05-14\part_000.txt' 
keys_from_jsonl = read_jsonl_and_return_keys(file_path)

# 打印每个JSON对象的键
for keys in keys_from_jsonl:
    print(keys)
#print(keys_from_jsonl[0])