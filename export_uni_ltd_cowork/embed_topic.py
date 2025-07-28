import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# from transformers import AutoModel
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd


from preprocess.dataloader import load_paper_data
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
        # device = select_device(self.use_gpu)
        device = 'cpu'
        # device = 'cuda'
        # wait_until_gpu_free(gpu_id=torch.cuda.device_count()-1, threshold=20, check_interval=20)
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
        retrieval_instruct = "Given a scientific domain as query, retrieve relevant papers that closely related to the domain. Note that the query should be a description and related keywords of the research field. The exact match of tokens does not matter, but the semantic meaning is crucial."
        self.move_model(select_device(self.use_gpu))
        query_embeddings = self.model.encode(queries, prompt=f"Instruct: {retrieval_instruct}\nQuery: ")
        self.move_model('cpu')
        return query_embeddings
    
    def embed_query_papers(self, queries: List[Dict[str, str]]):
        paper_string = [format_embedding_paper_string(query) for query in queries]
        self._init_model()
        # retrieval_instruct = ""
        self.move_model(select_device(self.use_gpu))
        # import pdb; pdb.set_trace()  # DEBUG paper_string
        query_embeddings = self.model.encode(paper_string, prompt=f"Instruct: 为检索与该论文内容相似的论文生成向量表示。\nQuery: ")
        self.move_model('cpu')
        return query_embeddings

    def embed_paper(self, documents: List[str]):
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
        self._init_model()
        self.move_model(select_device(self.use_gpu))
        document_embeddings = self.model.encode(documents)
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
            batch_embeddings = self.model.embed_paper([doc for _, doc in batch_docs])

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

    def search_by_paper(self, papers: List[Dict[str, str]], top_k=5000, sim_lower_bound=0.3) -> List[Tuple[str, float]]:
        query_embeddings = self.model.embed_query_papers(papers)
        if not isinstance(query_embeddings, np.ndarray):
            query_embeddings = query_embeddings.nunmpy()
            
        # import pdb; pdb.set_trace()
        # 结果累加用
        sim_scores = {}  # doc_id -> [similarities]

        for query in query_embeddings:
            similarities, indices = self.index.search(query.reshape(1, -1).astype('float32'), top_k * 2)
            for idx, sim in zip(indices[0], similarities[0]):
                if sim >= sim_lower_bound:
                    doc_data = self.metadata[idx]
                    sim_scores.setdefault(doc_data, []).append(sim)
        # import pdb; pdb.set_trace()
        # 聚合相似度（这里用平均值）
        aggregated = [(doc_data, float(np.mean(sims))) for doc_data, sims in sim_scores.items()]
        # 排序返回
        aggregated = sorted(aggregated, key=lambda x: x[1], reverse=True)[:top_k]
        
        # import pdb; pdb.set_trace()  # DEBUG results
        if self.if_clear:    
            self._clear()

        return aggregated

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

def format_embedding_paper_string(paper_data: Dict[str, str]) -> str:
    '''
    合并论文的各种信息为一个字符串
    '''
    # 生成用于与研究主题匹配检索的科学论文向量表示。
    output_str = "以下是论文信息："
    keys = [
        "title",
        "keywords",
        "paper journal/conference name",
        "paper subject",
        "paper project",
        "paper institution"
    ]
        
    for key in keys:
        output_str += f"{key}: {paper_data.get(key, '')}; "
    return output_str

def main_embed_paper(output_file='paper_embeddings.index'):
    engine = RetrievalEngine(os.path.join('embedded_data', output_file), embedder_obj=PaperEmbedding())
    
    # 创建 embedded_data 文件夹
    if not os.path.exists('embedded_data'):
        os.makedirs('embedded_data')

    root_dir = os.path.dirname(os.path.abspath(__file__))
    paper_data_map, author_paper_map, author_data_map = load_paper_data(os.path.join(root_dir, "data/2000-2024论文数据 20250214.xlsx"))
    
    engine.build_index({k: format_embedding_paper_string(v) for k, v in paper_data_map.items()})
    # store other dataset
    with open(os.path.join('embedded_data', 'embedding_support_data.pkl'), 'wb') as f:
        pickle.dump({
            'paper_data_map': paper_data_map,
            'author_paper_map': author_paper_map,
            'embedding_metadata': engine.metadata,
            'author_data_map': author_data_map
        }, f)
    

if __name__ == "__main__":
    main_embed_paper('paper_embeddings.index')
