import json
import faiss
import yaml
import numpy as np
import math
import os
import jieba
import pickle
import sys
import logging
import subprocess
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker

from .my_logger import logger
from .gpt import fuzzysearch_extractor_prompt,Variety_disease_extractor_prompt
from .gpt import GPT

class retriever(object):
    def __init__(self):
        # 此段代码是为了使得国内网络也能下载huggingface模型（也许没用），也可手动export HF_ENDPOINT=https://hf-mirror.com
        # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        # if os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com':
        #     logger.info('HF_ENDPOINT 设置为 "https://hf-mirror.com"')
        # self.init_huggingface()
        pass

    def init_huggingface(self):
        command = "export HF_ENDPOINT=https://hf-mirror.com"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

class BGE_retriever(retriever):
    def __init__(self,raw_file_path,ragdb_file_path=None,index_file_path=None):
        super().__init__()
        self.raw_file_path = raw_file_path
        save_name=os.path.splitext(os.path.basename(raw_file_path))[0]
        self.ragdb_file_path = ragdb_file_path if ragdb_file_path is not None else os.path.join(os.getcwd(),f're_data/{save_name}/{save_name}.ragdb')
        self.index_file_path = index_file_path if index_file_path is not None else os.path.join(os.getcwd(),f're_data/{save_name}/{save_name}.index')
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.knowledge=self.read_txt(self.raw_file_path)
        self.index=self.get_index()
        self.ragdb = self.get_ragdb()

    def get_ragdb(self):
        if os.path.exists(self.ragdb_file_path):
            return self.load_ragdb()
        else:
            return None

    def get_index(self):
        if os.path.exists(self.index_file_path):
            logger.info(f"存在索引，正在加载文件：{self.index_file_path}")
            index = faiss.read_index(self.index_file_path)
        else:
            logger.info("不存在索引，正在建立索引")
            index=self.build_index(self.knowledge)
        return index

    def save_ragdb(self,ragdb):
        logger.info(f"保存知识库为数据库文件，保存在{self.ragdb_file_path}下")
        with open(self.ragdb_file_path, 'wb') as f:
            pickle.dump(ragdb, f)

    def load_ragdb(self):
        with open(self.ragdb_file_path, 'rb') as f:
            ragdb = pickle.load(f)
        return ragdb

    def build_faiss_index(self,embeddings):
        dim=1024
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info(f'索引建立成功，正在保存索引文件，文件保存在{self.index_file_path}下')
        faiss.write_index(index, self.index_file_path)

        return index

    def read_txt(self,file_path):
        r"""
        将txt文件按行读取为列表
        """
        lines = []
        try:
            with open(file_path, 'r',encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print("文件未找到")
        except Exception as e:
            print("发生错误：", e)

        return lines

    def bge_embedding(self,sentences,return_dense=True,return_sparse=False,return_colbert_vecs=False):
        embeddings = self.model.encode(sentences,return_dense=return_dense,return_sparse=return_sparse,return_colbert_vecs=return_colbert_vecs)
        return embeddings

    def build_index(self,sentences,save_ragdb=False):
        logger.info("正在计算知识库向量")
        embeddings = self.bge_embedding(sentences)
        logger.info("计算完成，正在建立索引")
        dense_vecs=embeddings['dense_vecs']
        lexical_weights=embeddings['lexical_weights']
        colbert_vecs=embeddings['colbert_vecs']

        index=self.build_faiss_index(dense_vecs)
        if save_ragdb:
            text_emb_pairs = [{"text": sentences[i], "dense_vecs": dense_vecs[i], "lexical_weights": lexical_weights[i],
                               "colbert_vecs": colbert_vecs[i]} for i in range(len(sentences))]
            self.save_ragdb(text_emb_pairs)

        return index

    def search(self,querys,top_k=10,return_vecs_dict=True):
        knowledge=self.knowledge
        index=self.index
        if not isinstance(querys,list):
            querys=[querys]
        logger.info(f"正在计算问题向量")
        querys_embeddings=self.bge_embedding(querys)['dense_vecs']
        logger.info("计算完成，正在检索")
        D,I=index.search(querys_embeddings,top_k)
        logger.info("检索完成")
        results=[]
        for i in range(len(I)):
            results.append([[knowledge[idx] for idx in I[i]],D[i]])

        if return_vecs_dict:
            results_list = []
            for item in results:
                results_list.append([item[0], item[1].tolist()])
        return results_list

class BM25Param(object):
    def __init__(self, f, df, idf, length, avg_length, docs_list, line_length_list,k1=1.5, k2=1.0,b=0.75):
        """

        :param f:
        :param df:
        :param idf:
        :param length:
        :param avg_length:
        :param docs_list:
        :param line_length_list:
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param b:
        """
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"

class BM25_retriever(retriever):
    def __init__(self,raw_file_path,stop_words_path=None,index_file_path=None):
        super().__init__()
        self.raw_file_path = raw_file_path
        save_name=os.path.splitext(os.path.basename(raw_file_path))[0]
        self.stop_words_path = stop_words_path if stop_words_path is not None else os.path.join(os.getcwd(),f're_data/{save_name}/stop_words.txt')
        self.index_file_path = index_file_path if index_file_path is not None else os.path.join(os.getcwd(),f're_data/{save_name}/{save_name}.pkl')
        jieba.setLogLevel(log_level=logging.INFO)
        self.param = self._load_param()

    def _load_stop_words(self):
        if not os.path.exists(self.stop_words_path):
            raise Exception(f"system stop words: {self.stop_words_path} not found")
        stop_words = []
        with open(self.stop_words_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _load_param(self):
        self._stop_words = self._load_stop_words()
        if not os.path.exists(self.index_file_path):
            param = self._build_param()
        else:
            with open(self.index_file_path, 'rb') as reader:
                param = pickle.load(reader)
        return param

    def _build_param(self):

        def _cal_param(reader_obj):
            f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
            df = {}  # 存储每个词及出现了该词的文档数量
            idf = {}  # 存储每个词的idf值
            lines = reader_obj.readlines()
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list =[]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                words = [word for word in jieba.lcut(line) if word and word not in self._stop_words]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                f.append(tmp_dict)
                for word in tmp_dict.keys():
                    df[word] = df.get(word, 0) + 1
            for word, num in df.items():
                idf[word] = math.log(length - num + 0.5) - math.log(num + 0.5)
            param = self.BM25Param(f, df, idf, length, words_count / length, docs_list, line_length_list)
            return param

        if not os.path.exists(self.raw_file_path):
            raise Exception(f"system docs {self.raw_file_path} not found")
        with open(self.raw_file_path, 'r', encoding='utf8') as reader:
            param = _cal_param(reader)

        with open(self.index_file_path, 'wb') as writer:
            pickle.dump(param, writer)
        return param

    def _cal_similarity(self, words, index):
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            molecular = self.param.idf[word] * self.param.f[index][word] * (self.param.k1 + 1)
            denominator = self.param.f[index][word] + self.param.k1 * (1 - self.param.b +
                                                                       self.param.b * self.param.line_length_list[index] /
                                                                       self.param.avg_length)
            score += molecular / denominator
        return score

    def cal_similarity(self, query: str):
        """
        相似度计算，无排序结果
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        words = [word for word in jieba.lcut(query) if word and word not in self._stop_words]
        score_list = []
        for index in range(self.param.length):
            score = self._cal_similarity(words, index)
            score_list.append((self.param.docs_list[index], score))
        return score_list

    def search(self, querys,top_k=10,return_vec=True):
        """
        相似度计算，排序
        :param query: 待查询结果
        :return: [(doc, score), ..]
        """
        if not isinstance(querys, list):
            querys = [querys]
        results=[]
        for query in querys:
            result = self.cal_similarity(query)
            result.sort(key=lambda x: -x[1])
            result=result[:top_k]
            results.append(result)
        return results

class Topic_retriever(retriever):
    def __init__(self,raw_file_path,index_file_path=None):
        super().__init__()
        self.raw_file_path = raw_file_path
        save_name=os.path.splitext(os.path.basename(raw_file_path))[0]
        self.index_file_path = index_file_path if index_file_path is not None else os.path.join(os.getcwd(),f're_data/{save_name}/{save_name}.json')
        self.data_base=self.get_data_base()

    def get_data_base(self):
        if os.path.exists(self.index_file_path):
            logger.info("正在加载Topic数据库")
            return self.read_json_file(self.index_file_path)
        else:
            logger.info("未找到Topic数据库，正在构建")
            raw_data=self.read_txt(self.raw_file_path)
            name_to_context={}
            context_to_name={}
            i=1
            for line in raw_data:
                line=line.strip()
                first_space_index=line.index(' ')
                name=line[:first_space_index]
                context=line[first_space_index+1:]
                if name in name_to_context:
                    name_to_context[name].append(context)
                else:
                    name_to_context[name]=[context]
                context_to_name[context]=name
            data_base={'name_to_context':name_to_context,'context_to_name':context_to_name}
            self.save_list_of_dicts_to_json(data_base,self.index_file_path)
            return data_base

    def read_json_file(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def save_list_of_dicts_to_json(self,list_of_dicts, file_path):
        with open(file_path, "w") as file:
            json.dump(list_of_dicts, file, indent=4, ensure_ascii=False)

    def read_txt(self,file_path):
        r"""
        将txt文件按行读取为列表
        """
        lines = []
        try:
            with open(file_path, 'r',encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print("文件未找到")
        except Exception as e:
            print("发生错误：", e)

        return lines

    def search(self, querys,top_k=10,return_vec=True):
        if not isinstance(querys, list):
            querys = [querys]
        results=[]
        gpt_client=GPT()
        name_to_context=self.data_base['name_to_context']
        context_to_name=self.data_base['context_to_name']
        for query in querys:
            vd_prompt=Variety_disease_extractor_prompt.replace("{插入位置}",query)
            res1=gpt_client.gpt_chat(vd_prompt)
            vd_list=res1.split(",")
            vd_know=[]
            for v in vd_list:
                if v in name_to_context:
                    t_list=[v+" "+item for item in name_to_context[v]]
                    vd_know.extend(t_list)
            if not vd_know:
                fs_prompt=fuzzysearch_extractor_prompt.replace("插入位置",query)
                res2=gpt_client.gpt_chat(fs_prompt)
                fs_list=res2.split(",")
                fs_know=[]
                keys=list(context_to_name.keys())
                t2_list=[]
                for t in fs_list:
                    t2_list.extend([key for key in keys if t in key])
                t2_list=list(set(t2_list))
                for t2 in t2_list:
                    fs_know.append(context_to_name[t2]+' '+t2)
                vd_know.extend(fs_know)
            vd_know=vd_know[:top_k]
            result=[[item,0] for item in vd_know]
            results.append(result)
        return results

class Search_Warehouse(object):
    def __init__(self,data_base_name,default_config_path='./config/search_warehouse.yaml',config=None):
        self.data_base_name=data_base_name
        if config is None:
            logger.info("正在加载检索库配置文件")
            self.config=self.load_yaml_file(default_config_path)
        else:
            logger.info("已初始化检索器配置消息")
            self.config=config
        self.init_data_base(data_base_name)
        logger.info("成功初始化检索器")

    def init_data_base(self,data_base_name):
        logger.info("正在加载数据库")
        data_bases=self.config['data_bases']
        for data_base in data_bases:
            if data_base['name']==data_base_name:
                self.data_base=data_base
                break
        data_base=self.data_base
        if data_base['type']=='BGE_retriever':
            logger.info("检索器类型：BGE_retriever")
            try:
                raw_file_path=data_base['raw_file_path']
            except KeyError:
                raise Exception(f"请指定检索库{data_base_name}的原始文件：raw_file_path")
            ragdb_file_path=data_base['ragdb_file_path'] if data_base.get('ragdb_file_path',None) is not None else os.path.join(os.getcwd(),f're_data/{data_base_name}/{data_base_name}.ragdb')
            index_file_path=data_base['index_file_path'] if data_base.get('index_file_path',None) is not None else os.path.join(os.getcwd(),f're_data/{data_base_name}/{data_base_name}.index')
            self.retriever=BGE_retriever(raw_file_path,ragdb_file_path,index_file_path)
        if data_base['type']=='BM25_retriever':
            logger.info("检索器类型：BM25_retriever")
            try:
                raw_file_path=data_base['raw_file_path']
            except KeyError:
                raise Exception(f"请指定检索库{data_base_name}的原始文件：raw_file_path")
            stop_words_path=data_base['stop_words_path'] if data_base.get('stop_words_path',None) is not None else os.path.join(os.getcwd(),f're_data/{data_base_name}/stop_words.txt')
            index_file_path=data_base['index_file_path'] if data_base.get('index_file_path',None) is not None else os.path.join(os.getcwd(),f're_data/{data_base_name}/{data_base_name}.pkl')
            self.retriever=BM25_retriever(raw_file_path,stop_words_path, index_file_path)
        if data_base['type']=='Topic_retriever':
            logger.info("检索器类型：Topic_retriever")
            try:
                raw_file_path=data_base['raw_file_path']
            except KeyError:
                raise Exception(f"请指定检索库{data_base_name}的原始文件：raw_file_path")
            index_file_path=data_base['index_file_path'] if data_base.get('index_file_path',None) is not None else os.path.join(os.getcwd(),f're_data/{data_base_name}/{data_base_name}.json')
            self.retriever=Topic_retriever(raw_file_path,index_file_path)
        return

    def load_yaml_file(self,file_path):
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            return data
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except yaml.YAMLError as e:
            print("Error loading YAML file:", e)
            return None

    def search(self,querys,top_k=10,return_vecs_dict=True):
        return self.retriever.search(querys,top_k,return_vecs_dict)