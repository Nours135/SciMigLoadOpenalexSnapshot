# 读取解压缩的所有文件，并且存入本地数据库中
from decompose import scan_list, extract_certain_suffix
import json
import pymysql
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


def insert_add_and_edit(work_id, work_data, base_name, db, cursor):
    '''如何做到如果在数据库内就更新json，不在数据库内就插入新数据？？大概就try插入，报错就改成update'''
    if base_name == 'clinical':
        query = """
            INSERT INTO works_clinical2 (work_OpenAlex_id, work_info)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE work_info = VALUES(work_info);
            """
    elif base_name == 'physic':
        query = """
            INSERT INTO works_physic2 (work_OpenAlex_id, work_info)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE work_info = VALUES(work_info);
            """
    else:
        raise ValueError(f"invalid base_name {base_name}")
    
    values = (work_id, json.dumps(work_data))
    try:
        # 执行sql语句
        cursor.execute(query, values)
        # 提交到数据库执行
        db.commit()
    except Exception as e:
        # 如果发生错误则回滚
        with open("./works_report_new.txt", "a") as f:
            f.write(str(e) + "|" + str(work_id) +"\n")
        db.rollback()



if __name__ == '__main__':
    db = pymysql.connect(host='localhost',
                         port=3306,
                         user='localalex',
                         password='KLWDriver10086',
                         database='openalex')
    cursor = db.cursor()
    
    clinical_author_set = get_author_set(['clinical'])
    physic_author_set = get_author_set(['physics'])

    in_set_author_count = 0
    not_in_set_author_count = 0

    json_data_l = extract_certain_suffix(scan_list('works'), 'txt')
    # print(json_data_l)
    json_data_l = sorted(json_data_l, key=my_sort_file, reverse=False)  # 排序
    task = json_data_l[164:]
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
                if check_author_in(author_id_l, clinical_author_set):  # 是 则插入数据库
                    insert_add_and_edit(work_openalex_id, json_obj, 'clinical', db, cursor)
                if check_author_in(author_id_l, physic_author_set):
                    insert_add_and_edit(work_openalex_id, json_obj, 'physic', db, cursor)
                

            except Exception as err:
                if not os.path.exists('work\'s_author_info_parse_erros.txt'):
                    with open('work\'s_author_info_parse_erros.txt', 'w') as fp:
                        pass
                with open('work\'s_author_info_parse_erros.txt', 'a') as fp:
                        fp.write(f'{str(err)} | {work_openalex_id}')