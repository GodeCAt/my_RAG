import yaml
import os
import re

from .retriever import Search_Warehouse
from .chat_model import Model_LLM
from .my_logger import logger
from .gpt import GPT

def include(self,node):
    filename = os.path.join(os.getcwd(), 'config', node.value)
    with open(filename, 'r') as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)
yaml.add_constructor('!include', include)

class RAG_flow:
    def __init__(self, search_warehouse, model_llm, config_path='./config/rag_flow.yaml', post_llm_name='gpt'):
        self.config=self.load_yaml(config_path)
        logger.info(f"初始化检索器{search_warehouse}")
        self.search_warehouse=Search_Warehouse(search_warehouse, config=self.config['search_warehouse'])
        logger.info(f"初始化大模型{model_llm}")
        self.model_llm=Model_LLM(model_llm, config=self.config['model_llm'])
        if post_llm_name == 'gpt':
            self.post_llm = GPT()

    def load_yaml(self, file_name):
        logger.info("正在加载RAG_flow配置文件")
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding="utf-8") as fr:
                dict_obj = yaml.load(fr, Loader=yaml.FullLoader)
            return dict_obj
        else:
            raise FileNotFoundError('NOT Found YAML file %s' % file_name)

    def BM25_flow(self,query,post_process=True,threshold=5):
        re_result = self.search_warehouse.search(querys=query)[0]
        logger.info("BM25检索完成")
        if post_process:
            logger("正在进行后处理：1")
            return self.post_process_func1(query,re_result)
        else:
            logger("正在进行后处理：2")
            re_result = re_result[0]
            scores = re_result[1]
            num = 0
            for score in scores:
                if score > threshold:
                    num += 1
            citation = re_result[0][:num]
            scores = re_result[1][:num]
            citation=[citation, scores]
            return self.post_process_func2(query,citation)

    def BGE_flow(self,query,post_process=True,threshold=0.5):
        re_result = self.search_warehouse.search(querys=query)[0]
        logger.info("BGE检索完成")
        if post_process:
            logger("正在进行后处理：1")
            return self.post_process_func1(query, re_result)
        else:
            logger("正在进行后处理：2")
            re_result = re_result[0]
            scores = re_result[1]
            num = 0
            for score in scores:
                if score > threshold:
                    num += 1
            citation = re_result[0][:num]
            scores=re_result[1][:num]
            citation[citation,scores]
            return self.post_process_func2(query, citation)

    def Topic_flow(self,query,post_process=True):
        result = self.search_warehouse.search(querys=query)[0]
        knowledge_list = result
        logger.info("Topic检索完成")
        if post_process:
            logger("正在进行后处理：1")
            return self.post_process_func1(query,knowledge_list)
        else:
            logger("正在进行后处理：2")
            return self.post_process_func2(query,knowledge_list)

    def post_process_func1(self,query,result):
        knowledge_list = result
        useful_context = []
        logger.info("正在判断证据有效性")
        for knowlege in knowledge_list:
            context = knowlege[0]
            prompt = f"""判断以下的上下文是否能够回答问题，若上下文包含回答问题所需信息则输出'【是】',否则输出'【否】'
上下文：{context}
问题：{query}
回答："""
            response, mes = self.post_llm.chat(prompt)
            valid = 1 if "【是】" in response else 0
            if valid == 1:
                useful_context.append(context)
        results = []
        logger.info("正在生成有效证据回复")
        for context in useful_context:
            template = f"""你是问答任务的助手。
使用以下检索到的上下文来回答问题。
保持答案具体。
上下文：{context}
问题：{query}
回答："""
            res, mes = self.model_llm.chat(template)
            ans = {}
            ans['knowledge'] = context
            ans['answer'] = res
            results.append(ans)
        logger.info("正在进行最终判断")
        valid_ans = results
        if not valid_ans:
            final_answer_idx = None
        else:
            prompt = f"""给出问题以及一些检索上下文及回答，上下文是检索事实，选择其中能够最好回答问题的回答序号，最佳回答序号只有一个。
                    问题：{query}
                        """
            for index, ans in enumerate(valid_ans):
                prompt += f""""上下文{index + 1}：{ans['knowledge']}\n回答{index + 1}：{ans['answer']}\n\n"""
            prompt += "最佳答案序号为："
            response, mes = self.post_llm.chat(prompt)
            if len(re.findall(r'\d+', response)) == 0:
                final_answer_idx = None
            else:
                final_answer_idx = int(re.findall(r'\d+', response)[0]) - 1
        if final_answer_idx == None or len(results) <= final_answer_idx:
            res, mes = self.model_llm.chat(query)
            return res, []
        else:
            return results[final_answer_idx]['answer'], [results[final_answer_idx]['knowledge']]

    def post_process_func2(self,query,citation):
        citation=citation[0]
        if not citation:
            logger.info("知识库中无相关知识，进行普通生成")
            res, mes = self.model_llm.chat(query)
            return res, citation
        else:
            logger.info("知识库中包含相关知识，进行检索增强生成")
            prompt_template = """你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。回答需要考虑聊天历史。
以下是知识库：
{knowledge}
以上是知识库。
"""
            prompt = prompt_template.format(knowledge=citation)
            prompt = prompt + "问题：" + query
            res, mes = self.model_llm.chat(prompt)
            return res,citation