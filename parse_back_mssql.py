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

# AI_author_set = get_author_set(['AI'])
import pickle
with open('AI_Export/top_authors.pkl', 'rb') as fp:
    AI_author_set = pickle.load(fp)

# print(AI_author_set)
print(len(AI_author_set))

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



def consumer(task_queue, chunkCounter):
    data_dict = {
        'work_OpenAlex_id': [], 'work_info': []
    }
    data_length = 0
    while True:
        updates_f = task_queue.get()
        print(f'读取任务，队列剩余{task_queue.qsize()}')
        if updates_f is None:
            df = pd.DataFrame(data_dict)
            with chunkCounter.get_lock():
                cur_id = chunkCounter.value
                chunkCounter.value += 1
            df.to_csv(f'AI/chunk_{cur_id}')
            return

        for json_obj in tqdm(parse_openalex_snapshot_one_file(updates_f)):
            work_openalex_id = json_obj['id']
            try:
                authors_l = json_obj['authorships']
                author_id_l = []
                for author in authors_l:
                    author_id_l.append(author['author']['id'])  # 拿到了author list
                # 开始查验是否在set里面
                if check_author_in(author_id_l, AI_author_set):  # 是 则插入数据库
                    data_dict['work_OpenAlex_id'].append(work_openalex_id)
                    data_dict['work_info'].append(json_obj)
                    data_length += 1

            except Exception as err:
                pass
            if data_length > 10000:
                df = pd.DataFrame(data_dict)
                with chunkCounter.get_lock():
                    cur_id = chunkCounter.value
                    chunkCounter.value += 1
                df.to_csv(f'AI/chunk_{cur_id}')
                data_dict = {
                    'work_OpenAlex_id': [], 'work_info': []
                }
                data_length = 0
        # 结束后，最后的数据收尾
        if data_length > 0:
            df = pd.DataFrame(data_dict)
            with chunkCounter.get_lock():
                cur_id = chunkCounter.value
                chunkCounter.value += 1
            df.to_csv(f'AI/chunk_{cur_id}')
            data_dict = {
                'work_OpenAlex_id': [], 'work_info': []
            }
            data_length = 0


if __name__ == '__main__':
    if not os.path.exists('AI'): os.makedirs('AI')

    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    
    task_queue = multiprocessing.Queue()
    # https://zhuanlan.zhihu.com/p/649520663 根据这个修改为下面这个
    # mergeSort_task_queue = multiprocessing.Manager().Queue(-1)
    for item in json_data_l:
        task_queue.put(item) #放任务

    chunkCounter = multiprocessing.Value('i', 0)  # 计数器
    
    Consumers = []
    Consumers_count = 50
    for i in range(Consumers_count):
        Consumers.append(
            multiprocessing.Process(target=consumer, args=(task_queue, chunkCounter), daemon=True)
        )

    print('consumer 初始化成功')
    # producer来了，读取和排序

    for i in range(Consumers_count + 5):
        task_queue.put(None)

    for obj in Consumers:
        obj.start()

    for obj in Consumers:
        obj.join()

    print('结束')
    
