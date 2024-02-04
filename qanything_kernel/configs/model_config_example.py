import os
import logging
import uuid
from dotenv import load_dotenv
load_dotenv()

# LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.basicConfig(format=LOG_FORMAT)
# 获取项目根目录
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
print("UPLOAD_ROOT_PATH:", UPLOAD_ROOT_PATH)

# LLM streaming reponse
STREAMING = False

# PROMPT_TEMPLATE = """参考信息：
# {context}
# ---
# 我的问题或指令：
# {question}
# ---
# 请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
# 你的回复："""
SYSTEM_TEMPLATE="""
###系统名称
结构化深度信息解析系统
###系统描述
该系统旨在通过深度分析和结构化表达，准确回答用户的问题，特别注重引用准确的数据和报告来源，以及按照一定的逻辑顺序展开回答。
###工作原理
深度分析：仔细分析提供的参考信息，识别与问题直接相关的数据或报告。
逻辑结构：按照逻辑和重要性顺序组织回答，清晰地分解各个要点。
准确引用：对于每一个论点或结论，提供明确的来源引用，确保信息的可追溯性和可信度。
###回答策略
引言：使用“根据已知信息，可以得到以下答案：”作为回答的开头，明确指出回答基于提供的参考信息。
逐点回答：
对于每个要点，使用清晰的编号或者符号进行标记。
对于每个论点，提供具体的数据或报告支持，并注明来源。
结尾：简要总结回答的要点，强调其对问题的回答或意义。
语言风格：根据用户问题的语种进行回复，保持回答的一致性和专业性。
###回答格式
根据已知信息，可以得到以下答案：

<开头综述>，具体表现在以下几个方面：
1. 原因1...
-来源1
2. 原因2...
-来源2
3....
综上所述，<结尾总结>。
"""
PROMPT_TEMPLATE="""参考信息：
{context}
---
我的问题或指令：
{question}
---
请根据上述参考信息回答我的问题或回复我的指令。在回答时，请遵循以下指南：

1. 仔细筛选我提供的参考信息中与我的问题最相关的内容。
2. 保持回答忠实于原文，尽量简洁而全面，避免添加无关信息。
3. 使用与我的问题相同的语言进行回复。
4. 在回复中明确指出信息的出处，以便验证信息的准确性。

你的回复应该以“根据已知信息，可以得到以下答案：”开头，并按照以下结构组织内容：

- 首先概述回答的主要观点。
- 然后列出支持该观点的具体事实或数据，每一项都明确指出信息来源。
- 最后，总结上述信息，强调其对我的问题或指令的相关性和重要性。

请按照这种方式回答我的问题或回复我的指令。
"""
# For LLM Chat w/o Retrieval context
# PROMPT_TEMPLATE = """{question}"""

QUERY_PROMPT_TEMPLATE = """{question}"""

# 缓存知识库数量
CACHED_VS_NUM = 100

# 文本分句长度
SENTENCE_SIZE = 250

# 匹配后单段上下文长度
CHUNK_SIZE = 800

# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 100

# embedding检索的相似度阈值，归一化后的L2距离，设置越大，召回越多，设置越小，召回越少
VECTOR_SEARCH_SCORE_THRESHOLD = 1.1

# NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# print('NLTK_DATA_PATH', NLTK_DATA_PATH)

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = True

# MILVUS向量数据库地址
MILVUS_HOST_LOCAL = "127.0.0.1"
MILVUS_HOST_ONLINE = "127.0.0.1"
MILVUS_PORT = 19530
# MYSQL数据库地址
MYSQL_HOST_LOCAL = "127.0.0.1"
MYSQL_HOST_ONLINE = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '123456'
MYSQL_DATABASE = 'qanything'
# OPENAI 配置
OPENAI_API_KEY =
OPENAI_API_BASE =
OPENAI_API_MODEL_NAME =
OPENAI_API_CONTEXT_LENGTH = 4000
OPENAI_EMBED_MODEL_NAME =
OPENAI_EMBED_BATCH = 16
OPENAI_EMBED_MODEL_VERSION = "1.0"
OPENAI_EMBED_DIM = 1536 #768

OPENAI_RERANK_EMBED_MODEL_NAME =
OPENAI_RERANK_EMBED_BATCH = 16
OPENAI_RERANK_EMBED_MODEL_VERSION = "1.0"

llm_api_serve_model = os.getenv('LLM_API_SERVE_MODEL')
llm_api_serve_port = os.getenv('LLM_API_SERVE_PORT')
rerank_port = 8776

print("llm_api_serve_port:", llm_api_serve_port)
print("rerank_port:", rerank_port)


LOCAL_LLM_SERVICE_URL = f"localhost:{llm_api_serve_port}"
LOCAL_LLM_MODEL_NAME = llm_api_serve_model
LOCAL_LLM_MAX_LENGTH = 4096

LOCAL_RERANK_SERVICE_URL = f"localhost:{rerank_port}"
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 16
