import logging

# 创建一个logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# 创建一个文件处理程序
file_handler = logging.FileHandler('my_logger.log')
file_handler.setLevel(logging.INFO)

# 创建一个控制台处理程序
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个格式化程序
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理程序添加到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)