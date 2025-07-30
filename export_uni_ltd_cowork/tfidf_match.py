# 由于使用大模型会带来巨大的算力开销，所以还是在已有的部分数据的基础上，改到用 tfidf 吧

import json
from rank_bm25 import BM25Okapi
import pandas as pd
import os
import re
import pickle
from typing import List, Tuple
import numpy as np
from collections import Counter


def clean_text(text: str) -> str:
    # 去掉标点符号（中英文），只保留文字和空格
    text = re.sub(r'[^\w\s]', '', text)  # 英文标点
    text = re.sub(r'[\u3000-\u303F\uFF00-\uFFEF]', '', text)  # 中文全角符号
    text = text.lower()  # 转为小写
    return text



def load_bm25_corpus():
    '''
    从 3 万篇的论文的篇关摘 & 词的关键词，构造出 bm25 documents 的语料，然后新的论文就是 query
    
    returns: Tuple[paper_id_2_eco_code, paper_index_2_paper_id, paper_document_l]
    '''
    # load all paper orgin data
    paper_id_2_full_info = dict()
    with open('company_papers.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            paper_id_2_full_info[data['id']] = data

    # load all paper into a list
    paper_index_2_paper_id = list(paper_id_2_full_info.keys())    # index -> paper_id
    paper_document_l = []

    for paper_id in paper_index_2_paper_id:
        cur_paper_full_info = paper_id_2_full_info[paper_id]
        # 论文的篇关摘
        tmp = []
        if 'title' in cur_paper_full_info and cur_paper_full_info['title']:
            tmp += clean_text(cur_paper_full_info['title']).split(" ")  # List of tokens 
        # concat abstract
        if 'abstract_inverted_index' in cur_paper_full_info and cur_paper_full_info['abstract_inverted_index']:
            for token, inverted_index in cur_paper_full_info['abstract_inverted_index'].items():
                tmp += [clean_text(token)] * len(inverted_index)   
        # concat keywords
        if 'keywords' in cur_paper_full_info and cur_paper_full_info['keywords']:
            tmp += [item['keyword'] for item in cur_paper_full_info['keywords']]
        # concat concepts
        if 'concepts' in cur_paper_full_info and cur_paper_full_info['concepts']:
            tmp += [item['display_name'] for item in cur_paper_full_info['concepts']]
        # concat mesh
        if 'mesh' in cur_paper_full_info and cur_paper_full_info['mesh']:
            # import pdb; pdb.set_trace()
            tmp += [item['descriptor_name'] if item['qualifier_name'] is None else item['descriptor_name'] + ' ' + item['qualifier_name'] for item in cur_paper_full_info['mesh']]
        
        paper_document_l.append(tmp)
        # import pdb; pdb.set_trace()


    # load index 2 eco_code
    paper_id_2_eco_code = dict()
    df = pd.read_excel('eco_domain_match_res.xlsx')
    for _, row in df.iterrows():
        paper_id_2_eco_code[row['id']] = row['domain_id']
        
    # import pdb; pdb.set_trace()

    return paper_id_2_eco_code, paper_index_2_paper_id, paper_document_l

# def main_generate_document_obj(reload = False):
#     paper_id_2_eco_code, paper_index_2_paper_id, paper_document_l = load_bm25_corpus()

#     if not reload and os.path.exists('bm25_documents.json'):
#         with open('BM25Okapi_obj.pkl', 'rb') as f:
#             bm25 = pickle.load(f)
#         return bm25, paper_index_2_paper_id, paper_id_2_eco_code
    
#     # 生成 BM25 的 documents
#     bm25 = BM25Okapi(paper_document_l)
#     # 保存到文件
#     with open('BM25Okapi_obj.pkl', 'wb') as f:
#         pickle.dump(bm25, f)
#     return bm25, paper_index_2_paper_id, paper_id_2_eco_code

class Retriever():
    def __init__(self, reload = False):
        paper_id_2_eco_code, paper_index_2_paper_id, paper_document_l = load_bm25_corpus()
        print(f'Loaded {len(paper_document_l)} documents for BM25 retrieval.')
        self.paper_index_2_paper_id = paper_index_2_paper_id
        self.paper_id_2_eco_code = paper_id_2_eco_code

        if not reload and os.path.exists('bm25_documents.json'):
            with open('BM25Okapi_obj.pkl', 'rb') as f:
                self.bm25 = pickle.load(f)

        else:
            # 生成 BM25 的 documents
            self.bm25 = BM25Okapi(paper_document_l)
            # 保存到文件
            with open('BM25Okapi_obj.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # use mean similarity
        # self.eco_code_2_count = Counter(paper_id_2_eco_code.values())
        # import pdb ; pdb.set_trace()  # debug count stats


    def retrieve(self, tokenized_query: List[str], top_n=500):
        '''
        实际上是一个基于 BM25 相似度计算的 KNN
        '''
        doc_scores = self.bm25.get_scores(tokenized_query)   # array of scores, size [n_corpus]
        # get top n
        top_n_indices = np.argsort(doc_scores)[-top_n:][::-1]
        top_n_scores = doc_scores[top_n_indices]
        top_n_paper_ids = [self.paper_index_2_paper_id[i] for i in top_n_indices]
        top_n_eco_codes = [self.paper_id_2_eco_code[i] for i in top_n_paper_ids]
        # all eco_codes
        # all_eco_codes = list(set(top_n_eco_codes))
        result = dict()
        eco_code_2_count = Counter(top_n_eco_codes)
        assert len(top_n_scores) == len(top_n_eco_codes), "Top N scores and eco codes must have the same length."
        for score, eco_code in zip(list(top_n_scores), top_n_eco_codes):
            if eco_code not in result:
                result[eco_code] = 0
            result[eco_code] += score
        
        for eco_code in result:
            result[eco_code] /= eco_code_2_count[eco_code]
            
        import pdb; pdb.set_trace()
        return result   




if __name__ == '__main__':
    # main_generate_document_obj()
    retriever = Retriever(reload=True)
    query = "machine learning in healthcare"

    retriever.retrieve(query.split(" "))



