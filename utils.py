# 本文件主要用于提供一个函数，用来筛选work是否是我们需要的
# 主要基于两个逻辑，这个work的作者是否在给定的 author list内

import pandas as pd
from decompose import scan_list, extract_certain_suffix
from tqdm import tqdm
import os
import re

def my_sort_file(f_name):
    '''我怎么感觉有必要按照时间先后顺序，parse这些json文件？？不管了，反正不难写吧，嗯...
    输入：路径，authors\\updated_date=2023-10-07\\part_000.txt
    输出：一个可以用来排序的tuple: (2023, 10, 7, 0)
    '''
    numbers = [int(n) for n in re.findall(r'\d+', f_name)]
    # print(numbers)
    return numbers  

def get_lenth(f_name):
    '''再写一个函数，获取长度吧'''
    count = 0
    with open(f_name, 'r', encoding='utf-8') as fp:
        for line in fp:
            count += 1
    return count

disciplines2csv = {'clinical': 'journal_author_ids_clinical.csv',
                   'management': 'journal_author_ids_management.csv',
                   'physics': 'journal_author_ids_physics.csv',
                   'AI':'journal_author_ids_AI2.csv'}

def get_author_set(disciplines=['clinical']):
    '''给出学科关键词的列表，返回学科作者的列表'''
    author_set = set()
    for discip in disciplines:
        df = pd.read_csv(disciplines2csv[discip], header=0)
        author_id_l = list(df['author_id'])
        author_set.update(author_id_l)

    return author_set


def check_author_in(authors, author_set):
    '''输入作者列表 authors
    如果有一个在author_set，返回True
    否则，返回False'''
    for author in authors:
        if author in author_set:
            return True
    return False



if __name__ == '__main__':
    from parse import parse_openalex_snapshot_one_file
    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    # task = json_data_l[100:120] + json_data_l[300:320]  # 选了200个解析
    task = json_data_l
    task_len = len(task)

    clinical_author_set = get_author_set(['clinical'])
    in_set_author_count = 0
    not_in_set_author_count = 0

    for i, updates_f in enumerate(task):
        pybar = tqdm(total=get_lenth(updates_f), desc=f'{i+1}/{task_len}个update chunk的条记录')
        for json_obj in parse_openalex_snapshot_one_file(updates_f):
            pybar.update(1)
            work_openalex_id = json_obj['id']
            try:
                authors_l = json_obj['authorships']
                author_id_l = []
                for author in authors_l:
                    author_id_l.append(author['author']['id'])  # 拿到了author list
                # 开始查验是否在set里面
                if check_author_in(author_id_l, clinical_author_set):
                    in_set_author_count += 1
                else:
                    not_in_set_author_count += 1

            except Exception as err:
                if not os.path.exists('work\'s_author_info_parse_erros.txt'):
                    with open('work\'s_author_info_parse_erros.txt', 'w') as fp:
                        pass
                with open('work\'s_author_info_parse_erros.txt', 'a') as fp:
                        fp.write(f'{str(err)} | {work_openalex_id}')

            

    print('在author集合中的文献数量：', in_set_author_count)
    print('不在的数量：', not_in_set_author_count)