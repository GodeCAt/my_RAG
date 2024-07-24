import sys
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import LoraConfig,load_peft_weights,AutoPeftModelForCausalLM,PeftModel
from colorama import Fore, Style
import yaml

from .LLM import Baichuan2,Qwen1_5
from .my_logger import logger

class Model_LLM:
    def __init__(self,chat_model_name,default_config_path='./config/model_llm.yaml',config=None):
        self.chat_model_name = chat_model_name
        if config is None:
            logger.info("正在加载大模型配置文件")
            self.config=self.load_yaml_file(default_config_path)
        else:
            self.config=config
        logger.info(f"正在初始化模型{chat_model_name}")
        self.model_llm=self.init_model(chat_model_name)
        logger.info("成功初始化大模型")

    def init_model(self,chat_model_name):
        model_bases=self.config['model_bases']
        for model_base in model_bases:
            if model_base['name']==chat_model_name:
                self.model_base=model_base
                break
        if self.model_base['model_type']=='Baichuan2':
            model_llm=Baichuan2(self.model_base)
        if self.model_base['model_type']=='Qwen1_5':
            model_llm=Qwen1_5(self.model_base)
        return model_llm

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

    def chat(self,prompt,messages=[],stream=False):
        responses,messages=self.model_llm.chat(prompt,messages,stream)
        return responses,messages

    def conversation(self,messages=[],stream=True):
        self.model_llm.conversation(messages,stream)