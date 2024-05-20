import gzip
import os
from tqdm import tqdm
# 在自动下载完后自动解压


def scan_list(path):
    '''输入文件夹路径
    输出列表，包含文件夹里的所有文件
    递归查找'''
    try:
        l = os.listdir(path)
    except NotADirectoryError:
        return [path]
    data = []
    for item in l:
        data += scan_list(path + '\\' + item)
    return data


def count_type(l):
    re = {}
    for item in l:
        suffix = item.split('.')[-1]
        re[suffix] = re.get(suffix, 0) + 1
    return re


def extract_certain_suffix(l, suffix):
    re = []
    for item in l:
        if item.split('.')[-1].lower() in suffix:
            re.append(item)
    return re


def decompress_gz_files(file_list):
    '''自动解压缩'''
    for file_path in tqdm(file_list):
        if file_path.endswith('.gz'):
            # 构建解压后的文件名（移除.gz）
            output_file_path = file_path[:-3]
            # 打开.gz文件
            with gzip.open(file_path, 'rb') as f_in:
                # 读取.gz文件内容
                file_content = f_in.read()
                # 写入解压后的内容到新文件
                with open(output_file_path + '.txt', 'wb') as f_out:
                    f_out.write(file_content)
            #print(f"文件已解压：{file_path} -> {output_file_path}")
        #else:
            #print(f"跳过非.gz文件：{file_path}")

if __name__ == '__main__':
    # print(*scan_list('authors'), sep='\n')
    gz_l = scan_list('merged_authors')
    decompress_gz_files(gz_l)