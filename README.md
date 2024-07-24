# my_RAG

这是我自己写的RAG框架，不需要软件环境依赖，只需要Python环境，写这个项目的目的是为了能够方便进行代码的迁移。

## 文件架构说明

config：包含检索器配置文件，大模型配置文件以及RAG_flow配置文件

re_data：检索器资源保存位置

util：构建RAG_flow的工具，retriever.py文件对检索器进行定义，chat_model.py为大模型加载类，可以加载不同大模型，LLM.py定义不同大模型类，rag_flow.py定义自己的RAG_flow。