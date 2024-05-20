from decompose import scan_list, extract_certain_suffix
from parse import my_sort_file, parse_openalex_snapshot_one_file, get_lenth
from tqdm import tqdm



if __name__ == '__main__':
    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    task = json_data_l[100:200] + json_data_l[300:400]  # 选了200个解析
    task_len = len(task)

    id_set = set()
    duplicates_count = 0

    for i, updates_f in enumerate(task):
        pybar = tqdm(total=get_lenth(updates_f), desc=f'{i+1}/{task_len}个update chunk的条记录')
        for json_obj in parse_openalex_snapshot_one_file(updates_f):
            pybar.update(1)
            work_openalex_id = json_obj['id']
            if work_openalex_id in id_set:
                duplicates_count += 1   # 重复记1
            id_set.add(work_openalex_id)

    print('重复work id数量：', duplicates_count)
    print('全部的work id 数量：', len(id_set))
            