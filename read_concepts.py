from parse import parse_openalex_snapshot_one_file
from utils import scan_list, extract_certain_suffix, my_sort_file, get_lenth
from tqdm import tqdm
import json



def parse_concepts():
    # final_result = dict()
    final_result = []
    ID_set = set()
    duplicate_count = 0

    json_data_l = extract_certain_suffix(scan_list('concepts'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    # print(json_data_l)
    task_len = len(json_data_l)
    for i, updates_f in enumerate(json_data_l):
        pybar = tqdm(total=get_lenth(updates_f), desc=f'{i+1}/{task_len}个update chunk的条记录')
        for json_obj in parse_openalex_snapshot_one_file(updates_f):
            pybar.update(1)
            final_result.append(json_obj)
            id = json_obj['id']
            if id in ID_set:
                duplicate_count += 1
            ID_set.add(id)
            # print(json_obj.keys())
            # break
        # break
    with open('concepts/concepts.json', 'w', encoding='utf-8') as fp:
        json.dump(final_result, fp)
    print(f'重复数量{duplicate_count}')


if __name__ == '__main__':
    # 下面开始，利用concepts重建树形结构

    with open('concepts/concepts.json', 'r', encoding='utf-8') as fp:
        final_result = json.load(fp)
    
