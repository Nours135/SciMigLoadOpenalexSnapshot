import json
import requests as rq
import os
from tqdm import tqdm

# 本文件的功能是根据manifest文件下载所有的.gz文件

def read_manifest():
    fp = open('concepts\manifest', 'r', encoding='utf-8')
    manifest = json.load(fp)
    return manifest

def change_url(url):
    ''' 将这种格式的url s3://openalex/data/authors/updated_date=2023-07-30/part_000.gz
    转化成可以直接下载的url：https://openalex.s3.amazonaws.com/data/authors/updated_date%3D2023-09-08/part_000.gz
    '''
    url_parts = url.split('openalex')
    new_url = 'https://openalex.s3.amazonaws.com' + url_parts[-1]
    new_url = new_url.replace('=', '%3D')
    # print(new_url)
    return new_url

def ensure_directory_for_file(relative_path):
    # 分离出路径中的目录部分
    directory = os.path.dirname(relative_path)
    # 检查目录是否存在
    if not os.path.exists(directory):
        # 如果目录不存在，创建它
        os.makedirs(directory)
        # print(f"已创建目录：{directory}")
        return 1
    else:
        return 0
        # print(f"目录已存在：{directory}")



def downloaded_entries(url):
    '''根据改变后的url下载文件，我就部分文件夹了'''
    res = rq.get(url)
    filename = url.replace('https://openalex.s3.amazonaws.com/data/', '')  # 去掉这些所有的
    filename = filename.replace('%3D', '=')
    if res.status_code == 200:
        # 打开一个文件用于写入
        ensure_directory_for_file(filename)
        with open(filename, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                # 如果有数据就写入文件
                if chunk:
                    f.write(chunk)
        # print(f"下载完成: {filename}")
    else:
        print("错误: 无法下载文件")


if __name__ == '__main__':
    manifest = read_manifest()
    for entity in tqdm(manifest['entries']):
        url = entity['url']
        downloaded_entries(change_url(url))