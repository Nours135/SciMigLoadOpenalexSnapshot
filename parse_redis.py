# 读取解压缩的所有文件，并且存入本地数据库中
from decompose import scan_list, extract_certain_suffix
import json
# import pymysql
import redis
from tqdm import tqdm
from utils import check_author_in, get_author_set, my_sort_file, get_lenth
import os

'''
CREATE TABLE `works`(
    `work_OpenAlex_id` varchar(100) NOT NULL,
    `work_info` longtext,
    PRIMARY KEY (`work_OpenAlex_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
'''





def parse_openalex_snapshot_one_file(f_name):
    '''这个txt文件，应该是每一行是一个json obj'''
    with open(f_name, 'r', encoding='utf-8') as fp:
        for line in fp:
            # 处理每一行的文本内容，这里可以根据需要进行操作
            yield json.loads(line)


def insert_add_and_edit(work_id, work_data, r):
    '''如何做到如果在数据库内就更新json，不在数据库内就插入新数据？？大概就try插入，报错就改成update'''

    # values = (work_id, json.dumps(work_data))
    try:
        # 执行sql语句
        r.set(work_id, json.dumps(work_data))

    except Exception as e:
        # 如果发生错误则回滚
        with open("./works_report_new.txt", "a") as f:
            f.write(str(e) + "|" + str(work_id) +"\n")




if __name__ == '__main__':
    r = redis.Redis(host='localhost', port=6379)
    print('redis 连接成功')
    '''db = pymysql.connect(host='localhost',
                         port=3306,
                         user='localalex',
                         password='KLWDriver10086',
                         database='openalex')
    cursor = db.cursor()'''
    
    all_author_set = get_author_set(['clinical', 'physics'])
    # physic_author_set = get_author_set(['physics'])

    in_set_author_count = 0
    not_in_set_author_count = 0

    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    task = json_data_l[258:]  # 先整200后的
    task_len = len(task)

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
                if check_author_in(author_id_l, all_author_set):  # 是 则插入数据库
                    insert_add_and_edit(work_openalex_id, json_obj, r)

                

            except Exception as err:
                if not os.path.exists('work\'s_author_info_parse_erros.txt'):
                    with open('work\'s_author_info_parse_erros.txt', 'w') as fp:
                        pass
                with open('work\'s_author_info_parse_erros.txt', 'a') as fp:
                        fp.write(f'{str(err)} | {work_openalex_id}')