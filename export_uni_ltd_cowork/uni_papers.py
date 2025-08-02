
import json
import os
import sys
from typing import Generator, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import pandas as pd

from find_company_paper import is_CN, dump_fpath_generator, is_company, is_education, load_one_dump_file
from tfidf_match import Retriever, get_tokenized_text_from_full_openalex_json


# 需要获取一个表格，所有中国大学的论文的
    #  论文id，所属机构（名称和id），经济领域（作为技术领域），被引数量，和其他meta
# 这样就可以计算出 /rho_{i, j, u} 的分母了
# 也可以用来计算 /gamma 了

FIELDS = [
    'openalex_id',
    'institution_name', 'institution_id',
    'eco_code', 'eco_code_name',
    'citation_count',
    'title',
    'publication_date'
]
ECO_CODE_2_NAME = dict()
df = pd.read_csv('extracted_categories_no_desc.csv', header=0)
for index, row in df.iterrows():
    ECO_CODE_2_NAME[row['Category']] = row['Description']
# ECO_CODE_2_NAME['N/A'] = 'N/A'  # add a default value for N/A
del df

def add_quote(item: str):
    return f'"{item}"'


def get_first_author_institution(item: Dict[str, Any]) -> Tuple[str, str]:
    '''
    获取第一作者的机构名称和机构id
    '''
    for author_d in item.get('authorships', []):
        if author_d['author_position'] == 'first':
            if 'institutions' in author_d and len(author_d['institutions']) > 0:
                institution = author_d['institutions'][0]
                return institution['display_name'], institution['id']        
    return 'N/A', 'N/A'


def process_single_dump(file_path: str) -> Generator[Dict[str, Any], None, None]:
    '''
    处理单个 dump 文件，提取出需要的信息
    '''
    print(f"Processing {file_path}")
    retriever = Retriever(reload=False)
    results = []
    for item in load_one_dump_file(file_path):
        for author_d in item.get('authorships', []):
            if author_d['author_position'] == 'first':
                if is_CN(author_d) and not is_company(author_d):  # only process Chinese papers
                    tmp = dict()
                    query = get_tokenized_text_from_full_openalex_json(item)
                    
                    tmp['openalex_id'] = item['id']
                    ins_name, ins_id = get_first_author_institution(item)
                    tmp['institution_name'] = ins_name
                    tmp['institution_id'] = ins_id

                    tmp['title'] = item.get('title', 'N/A')
                    tmp['publication_date'] = item.get('publication_date', 'N/A')
                    tmp['citation_count'] = item.get('cited_by_count', 0)

                    response = retriever.retrieve(query, top_n=100)  # retrieve the first sampled paper
                    tmp['eco_code'] = response[0][0] if response else 'N/A'
                    tmp['eco_code_name'] = ECO_CODE_2_NAME.get(tmp['eco_code'], 'N/A')
                    results.append(tmp)
                    # import pdb; pdb.set_trace()  # debug sampled papers

    return results



def main_get_all_CN_university_papers(folder, max_workers: int = 5):
    file_paths = [item for item in dump_fpath_generator(folder)][::-1]  #[150:156]   # DEBUG 50 files for testing
    output_file = 'university_papers.csv'
    out_fp = open(output_file, 'w', encoding='utf-8')
    
    
    out_fp.write(','.join(FIELDS) + '\n')


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_dump, arg): arg for arg in file_paths}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(file_paths), desc="Processing files"):
            try:
                result = future.result()  # 获取单个任务结果
                for item in result:
                    to_write_str = ','.join([
                        add_quote(item.get('openalex_id', 'N/A')),
                        add_quote(item.get('institution_name', 'N/A')),
                        add_quote(item.get('institution_id', 'N/A')),
                        add_quote(item.get('eco_code', 'N/A')),
                        add_quote(item.get('eco_code_name', 'N/A')),
                        str(item.get('citation_count', 0)),
                        add_quote(item.get('title', 'N/A')),
                        add_quote(item.get('publication_date', 'N/A'))
                    ])
                    out_fp.write(to_write_str + '\n')
                
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    out_fp.close()
    print(f"Results saved to {output_file}")



if __name__ == '__main__':
    main_get_all_CN_university_papers('../works', max_workers=13)


    # test single
    # file_path = '../works/updated_date=2025-04-20/part_000.txt'
    # single_results = process_single_dump(file_path)