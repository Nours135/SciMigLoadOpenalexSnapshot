# 读取解压缩的所有文件，并且存入本地数据库中
from decompose import scan_list, extract_certain_suffix
import json
import re
#import pymysql
# import pymssql
import pandas as pd
from tqdm import tqdm
from utils import check_author_in, get_author_set, my_sort_file, get_lenth
import os
import multiprocessing
import pickle

AI_author_set = get_author_set(['AI'])

def my_sort_file(f_name):
    '''我怎么感觉有必要按照时间先后顺序，parse这些json文件？？不管了，反正不难写吧，嗯...
    输入：路径，authors\\updated_date=2023-10-07\\part_000.txt
    输出：一个可以用来排序的tuple: (2023, 10, 7, 0)
    '''
    numbers = [int(n) for n in re.findall(r'\d+', f_name)]
    # print(numbers)
    # print(numbers)
    return numbers  


def parse_openalex_snapshot_one_file(f_name):
    '''这个txt文件，应该是每一行是一个json obj'''
    with open(f_name, 'r', encoding='utf-8') as fp:
        for line in fp:
        # 处理每一行的文本内容，这里可以根据需要进行操作
            yield json.loads(line)

# UPDATE 修改这个函数，变成除了source id 外，还需要根据名称匹配
def extract_source_work_author(updates_f):
    work_id_set = set()
    author_id_set = set()
    for json_obj in tqdm(parse_openalex_snapshot_one_file(updates_f)):
        # work_openalex_id = json_obj['id']
        try:
            # authors_l = json_obj['authorships']
            # author_id_l = []
            # for author in authors_l:
            #     author_id_l.append(author['author']['id'])  # 拿到了author list
            sourceInfo = json_obj['primary_location']['source']
            sid = sourceInfo['id']
            if sid in SOURCE_SET:
                work_id = json_obj['id']
                # print(work_id)
                authors_l = json_obj['authorships']
                # print(json_obj.keys())
                author_id_l = []
                for author in authors_l:
                    author_id_l.append(author['author']['id'])  # 拿到了author list
                # print(set(author_id_l))
                work_id_set.add(work_id)
                author_id_set = author_id_set | set(author_id_l)
        
        except Exception as err:
            pass
    return work_id_set, author_id_set

def merge_dict(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] = d1[k] | v
        else:
            d1[k] = v
    return d1

df = pd.read_excel('AI_Export/AI_sourceID.xlsx')
df['Source ID'] = df['Source ID'].apply(lambda x: 'https://openalex.org/' + x.replace('s', 'S'))
SOURCE_SET = set(df['Source ID'].tolist())
print(SOURCE_SET)

if __name__ == '__main__':
    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    
    # task_queue = multiprocessing.Queue()
    # https://zhuanlan.zhihu.com/p/649520663 根据这个修改为下面这个
    # mergeSort_task_queue = multiprocessing.Manager().Queue(-1)
    # for item in json_data_l:
    #     task_queue.put(item) #放任务

    chunkCounter = multiprocessing.Value('i', 0)  # 计数器
    
    pool = multiprocessing.Pool(50)
    
    count = 1
    work_id_set, author_id_set = set(), set()
    for cur_work_id_set, cur_author_id_set in pool.imap(extract_source_work_author, json_data_l):
        count += 1
        print(count)
        work_id_set = work_id_set | cur_work_id_set
        author_id_set = author_id_set | cur_author_id_set

    print(len(work_id_set))
    print(len(author_id_set))

    with open('AI_Export/top_works.pkl', 'wb') as fp:
        pickle.dump(file=fp, obj=work_id_set)

    with open('AI_Export/top_authors.pkl', 'wb') as fp:
        pickle.dump(file=fp, obj=author_id_set)

    # extract_source_work_author(r'works\updated_date=2023-09-02\part_000.txt')





    
