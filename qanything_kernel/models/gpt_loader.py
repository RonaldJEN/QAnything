from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from qanything_kernel.configs.model_config import OPENAI_API_KEY,OPENAI_API_BASE,OPENAI_API_MODEL_NAME
import os
from openai import OpenAI

# 添加代理
os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"


gpt_llm = ChatOpenAI(model=OPENAI_API_MODEL_NAME,
                     openai_api_key=OPENAI_API_KEY,
                     openai_api_base=OPENAI_API_BASE,
                     temperature=1)
gpt_client = OpenAI(api_key=OPENAI_API_KEY,
                 base_url=OPENAI_API_BASE,
                 timeout=120,
                 max_retries=1)

# Test
# print(gpt_llm.predict('hello'))
# print(gpt_client.embeddings.create(input = ['hello'], model='text-embedding-ada-002').data[0].embedding)