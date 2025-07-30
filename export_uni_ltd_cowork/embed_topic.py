
# 这部分的环境，可以用 CAR
# query: 国民经济分类
# document: paper topic

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# from transformers import AutoModel
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import pandas as pd
import json


# from preprocess.dataloader import load_paper_data  # not used
from utils.gpu_utils import wait_until_gpu_free, select_device, clear_gpu_memory

class PaperEmbedding:
    def __init__(self, use_gpu: bool = True):
        self.model = None
        if use_gpu:
            self.use_gpu = 'gpu'
        else:
            self.use_gpu = 'cpu'
        
    def _init_model(self):
        '''self.model 改为 lazy init，减少显卡占用时间，避免冲突'''
        if self.model is not None:  # 已经 init 过了
            return
        device = select_device(self.use_gpu)
        # device = 'cpu'
        # device = 'cuda'
        wait_until_gpu_free(gpu_id=torch.cuda.device_count()-1, threshold=20, check_interval=20)
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True,  device=device, model_kwargs={'torch_dtype': torch.float16})
        # In case you want to reduce the maximum length:
        self.model.max_seq_length = 1024
        # print(self.model.device)

        # import pdb; pdb.set_trace()  # DEBUG model init, 是否用上了两个 gpu

    def embed_query(self, queries: List[str]):
        '''
        输入：
            queries: 一个字符串列表，每个字符串代表一个查询
        输出：
            query_embeddings: 一个numpy数组，每个元素代表一个查询的嵌入向量
        '''
        # prompt name 的设置来源于 huggingface 上官方提供的 example，设置的原因详见以下两个链接
        # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
        # https://www.sbert.net/examples/training/prompts/README.html
        self._init_model()
        retrieval_instruct = "Given a economic domain as query, retrieve relevant papers that closely related to the domain. Note that the query should be the name of the economic field. The exact match of tokens does not matter, but the semantic meaning is crucial."
        self.move_model(select_device(self.use_gpu))
        # import pdb; pdb.set_trace()  # DEBUG queries
        query_embeddings = self.model.encode(queries, prompt=f"Instruct: {retrieval_instruct}\nQuery: ", batch_size=4, show_progress_bar=True)
        self.move_model('cpu')
        return query_embeddings
    

    def embed_paper(self, documents: List[str], move_model = True):
        '''
        输入：
            documents: 一个字符串列表，每个字符串代表一个文档
        输出：
            document_embeddings: 一个numpy数组，每个元素代表一个文档的嵌入向量
        '''
        if len(documents) == 0:
            print("No documents to embed")
            return None
        if len(documents) > 20000:
            print(f"Too many documents to embed ({len(documents)} documents), please reduce the number of documents")
            return None
        if move_model:
            self._init_model()
            self.move_model(select_device(self.use_gpu))
        document_embeddings = self.model.encode(documents)
        if move_model:
            self.move_model('cpu')
            clear_gpu_memory()
        return document_embeddings
    
    def move_model(self, to_device):
        # pass # currently do nothing
        print(f'model moved to: {to_device}')
        device_str = str(to_device)

        # assert to_device in ['cpu', 'cuda', 'cuda:0', 'cuda:1'], "to_device should be either 'cpu' or 'cuda'"
        if device_str == 'cpu':
            self.model.to(to_device)
        elif device_str.startswith('cuda'):
            wait_until_gpu_free(gpu_id=int(device_str[-1]), threshold=20, check_interval=20)
            self.model.to(to_device)
        # AttributeError: 'torch.device' object has no attribute 'startswith'



class RetrievalEngine:
    def __init__(self, index_file: str = 'faiss_index.index', embedder_obj: PaperEmbedding = None, metadata: List[str] = None, if_clear: bool = False, lazy_init: bool = False):
        # 加载预训练的模型
        self.model = embedder_obj
        self.embedding_dim = 3584       #  https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
        self.index_file = index_file

        # 如果索引文件已存在，则加载索引
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            if metadata is None:
                raise ValueError("Metadata should be provided when index file exists")
            else:
                self.metadata = metadata

        else:
            # 返回的是余弦相似度，并且按照相似度进行了降序排列，https://github.com/facebookresearch/faiss/issues/2343
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # 使用内积来计算余弦相似度
            # 初始化一个空的列表用于存储每个嵌入的元数据（例如，字符串标识）
            self.metadata = []

        self.if_clear = if_clear  # 表示检索完后是否会马上清理模型的显存
        self.lazy_init = lazy_init  # 表示是否 lazy init 模型
        if not self.lazy_init:    # 不 lazy init 的话，则直接在 engine 处就初始化模型
            self.model._init_model()

    def build_index(self, documents: Dict[str, str], batch_size: int = 25):
        """
        逐步将文档嵌入加入到索引中，每次处理一个批次。
        并为每个嵌入存储一个字符串标识。
        """
        # 计算文档的嵌入
        print("Computing embeddings...")
        documents = list(documents.items())
        # import pdb; pdb.set_trace()  # DEBUG documents
        
        # 为每个嵌入存储标识（例如，文档索引）
        print("Building index with metadata...")
        for i in tqdm(range(0, len(documents), batch_size), desc="Building index", total=len(documents) // batch_size, unit="batch"):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = self.model.embed_paper([doc for _, doc in batch_docs], move_model=False)

            # 将批次嵌入加入索引
            self.index.add(batch_embeddings.astype('float32'))  # shape [n_docs, 3584]

            # 为批次中的每个嵌入存储元数据
            self.metadata.extend([name for name, _ in batch_docs])

            # import pdb; pdb.set_trace()  # DEBUG batch iter
            if i % 10 == 0:
                # print(f"Processed {i} documents")
                faiss.write_index(self.index, self.index_file)

        # 存储索引到文件
        faiss.write_index(self.index, self.index_file)
        print("Index saved.")


    def search(self, query: str, top_k=5000, sim_lower_bound=0.4) -> List[Tuple[str, float]]:
        # 获取查询的嵌入
        query_embedding = self.model.embed_query([query])

        # 归一化查询嵌入
        # query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # 使用 FAISS 进行查询，返回距离和索引
        print(f"Searching for top {top_k} similar documents...")
        # 返回的是余弦相似度，并且按照相似度进行了降序排列，https://github.com/facebookresearch/faiss/issues/2343
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # import pdb; pdb.set_trace()  # DEBUG searched results
        # 输出相似文档及其元数据
        results = []
        for i in range(top_k):
            if similarities[0][i] < sim_lower_bound and i > 500:  # 拿 top 100 兜底
                continue
            doc_index = indices[0][i]
            sim = similarities[0][i]
            results.append((self.metadata[doc_index], float(sim)))   # 从 numpy float 转为python float
        # import pdb; pdb.set_trace()  # DEBUG results
        if self.if_clear:    
            self._clear()

        return results
    
    def _clear(self):
        '''清空显存'''
        # self.model = None
        clear_gpu_memory()

    def parse_results(self, results: List[Tuple[str, float]]) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
        '''解析 search 方法的返回结果
        输出为 excel 方便查看
        返回两个数据：
            1. 一个 pandas DataFrame，包含所有论文结果的作者、论文标题和相似度
            2. 一个字典，键为作者，值为该作者的论文列表（Tuple[title, scores]）
        '''
        output_df_list = []
        output_author_paper_map = dict()
        for item in results:
            doc_index = item[0]
            sim = item[1]
            splited = doc_index.split('@@@')
            if len(splited) != 3: continue
            # import pdb; pdb.set_trace()
            output_df_list.append({
                'first_author': splited[0],
                'last_author': splited[1],
                'title': splited[2],
                'similarity': sim
                })
            if splited[0] not in output_author_paper_map:
                output_author_paper_map[splited[0]] = []
            output_author_paper_map[splited[0]].append({
                'title': splited[2], 'score': sim, 'doc_index': doc_index
            })
            if splited[1] == splited[0]:
                continue  # 如果第一作者和通讯作者相同，则不重复添加到 output_author_paper_map 中

            if splited[1] not in output_author_paper_map:
                output_author_paper_map[splited[1]] = []
            output_author_paper_map[splited[1]].append({
                'title': splited[2], 'score': sim, 'doc_index': doc_index
            })

        # import pdb; pdb.set_trace()  # DEBUG output_df_list
        return pd.DataFrame(output_df_list), output_author_paper_map

def format_embedding_paper_string(paper_data: List[Any]) -> List[str]:
    '''
    合并论文的各种信息为一个字符串
    '''
    # 生成用于与研究主题匹配检索的科学论文向量表示。
    output_str = "以下是论文信息："
    keys = [
        "concepts",
        "keywords"
    ]
    # import pdb; pdb.set_trace()  # DEBUG paper_data
    output_str += f"title: {paper_data.get('title', '')}; "
    items = [d['display_name'] for d in paper_data.get('concepts', [])]
    output_str += f"concepts: {', '.join(items)}; "
    items = [d['keyword'] for d in paper_data.get('keywords', [])]
    output_str += f"keywords: {', '.join(items)}; "
    # import pdb; pdb.set_trace()  # DEBUG paper_data

    return output_str


def load_economic_domain_data(file_path: str) -> Dict[str, str]:
    df = pd.read_csv(file_path, header=0, encoding='utf-8')
    # import pdb; pdb.set_trace()  # DEBUG df
    return {row['Category']: row['Description'] for _, row in df.iterrows()}


def load_paper_data(file_path: str) -> Dict[str, Dict[str, str]]:
    '''input file is a json list'''
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):  
            data.append(json.loads(line.strip()))
    # import pdb; pdb.set_trace()  
    paper_data_map = {item['id']: item for item in data}
    return paper_data_map

def main_embed_paper(output_file='paper_embeddings.index'):
    engine = RetrievalEngine(os.path.join('embedded_data', output_file), embedder_obj=PaperEmbedding())
    
    # 创建 embedded_data 文件夹
    if not os.path.exists('embedded_data'):
        os.makedirs('embedded_data')

    root_dir = os.path.dirname(os.path.abspath(__file__))
    paper_data_map = load_paper_data("company_papers.jsonl")
    # author_paper_map = dict()
    
    # import pdb; pdb.set_trace()  

    engine.build_index({k: format_embedding_paper_string(v) for k, v in paper_data_map.items()})
    # store other dataset
    with open(os.path.join('embedded_data', 'embedding_support_data.pkl'), 'wb') as f:
        pickle.dump({
            'paper_data_map': paper_data_map,
            'embedding_metadata': engine.metadata,
        }, f)


def load_economic_domain_data(file_path: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(file_path, header=0, encoding='utf-8')
    # import pdb; pdb.set_trace()  # DEBUG, fields 
    data = []
    for _, row in df.iterrows():
        if pd.isna(row['Category']) or pd.isna(row['Name']):
            continue
        data.append((row['Category'], row['Name'], f"Economic domain name: {row['Name']}, some related techniques (not exclusive): {row['Description']}"))
    # import pdb; pdb.set_trace()
    return data


def main_embed_queries():
    embeding_model = PaperEmbedding()
    
    # 创建 embedded_data 文件夹
    if not os.path.exists('embedded_data'):
        os.makedirs('embedded_data')

    # 加载经济领域数据
    economic_domain_data = load_economic_domain_data("extracted_categories.csv")

    # 构建索引
    domain_embedding = embeding_model.embed_query([full_desc for code, name, full_desc in economic_domain_data])

    # store other dataset
    # import pdb; pdb.set_trace()  # DEBUG economic_domain_data
    with open(os.path.join('embedded_data', 'economic_domain_embedding.pkl'), 'wb') as f:
        pickle.dump({
            'economic_domain_data': economic_domain_data,
            'economic_domain_embedding': domain_embedding,
        }, f)


def main_find_economic_domain():
    '''查找论文及其对应的经济领域'''
    # 加载经济领域的嵌入
    with open(os.path.join('embedded_data', 'economic_domain_embedding.pkl'), 'rb') as f:
        economic_domain_data = pickle.load(f)
    print('economic_domain_embedding.pkl loaded')
    
    # 加载 faiss index & paper metadata
    index = faiss.read_index(os.path.join('embedded_data', 'paper_embeddings.index'))
    with open(os.path.join('embedded_data', 'embedding_support_data.pkl'), 'rb') as f:  
        paper_meta = pickle.load(f)
    # import pdb; pdb.set_trace()  # DEBUG paper_data_map
    print('embedding_support_data.pkl loaded')
    
    
    # 读取 faiss index 到 numpy
    index_np = index.reconstruct_n(0, index.ntotal)  # shape: (n_papers, d)

    # 计算余弦相似度
    query_embeddings = economic_domain_data['economic_domain_embedding']  # shape [n_domains, d]
    similarities = np.dot(index_np, query_embeddings.T)  # shape [n_papers, n_domains]
    
    print('similarities computed')

    # find best matched domain for each paper
    best_matched_domains = np.argmax(similarities, axis=1)  # shape
    best_matched_scores = np.max(similarities, axis=1)  # shape

    # format output excel
    df_data = []
    keys = ['id', 'doi', 'title', 'publication_date', 'concepts', 'keywords']
    for i in range(len(best_matched_domains)):  # for each paper 
        paper_id = paper_meta['embedding_metadata'][i]
        paper_full_info = paper_meta['paper_data_map'][paper_id]
        domain_id, domain_name, full_desc = economic_domain_data['economic_domain_data'][best_matched_domains[i]]
        score = best_matched_scores[i]
        
        tmp = {
            'domain_id': domain_id,
            'domain_name': domain_name,
            'similarity_score': score
        }

        # import pdb; pdb.set_trace()  # DEBUG paper_full_info

        for key in keys:
            if key in paper_full_info:
                tmp[key] = paper_full_info[key]
            else:
                tmp[key] = 'N/A'
        # process authorships
        authorships = paper_full_info.get('authorships', [])
        first_author = []
        other_authors = []
        for author in authorships:
            if author.get('author_position', '') == 'first':
                first_author.append(author)
            else:
                other_authors.append(author)

        tmp['first_author'] = json.dumps(first_author) if first_author else 'N/A'
        tmp['other_authors'] = json.dumps(other_authors) if other_authors else 'N/A'

        # import pdb; pdb.set_trace()  # DEBUG paper_full_info
        df_data.append(tmp)
    
    print('excel generated')
    df = pd.DataFrame(df_data, columns=['domain_id', 'domain_name', 'similarity_score'] + keys + ['first_author', 'other_authors'])
    # import pdb; pdb.set_trace()
    df.to_excel('eco_domain_match_res.xlsx')

if __name__ == "__main__":
    # testing
    # data = load_paper_data('company_papers.jsonl')    
    # format_embedding_paper_string(data[0])

    # main_embed_paper('paper_embeddings.index')   # engine.metadata 就是对应的东西 
    main_embed_queries()    # 检查了下，index 的顺序可以保证
    main_find_economic_domain()