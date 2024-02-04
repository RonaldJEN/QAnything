import time
from typing import List
from qanything_kernel.models.gpt_loader import gpt_client
import numpy as np
from scipy.spatial.distance import cosine
from qanything_kernel.configs.model_config import OPENAI_RERANK_EMBED_MODEL_NAME, OPENAI_RERANK_EMBED_BATCH


class LocalRerankBackend:
    def __init__(self):
        self.model_name = OPENAI_RERANK_EMBED_MODEL_NAME
        self.overlap_tokens = 80
        self.spe_id = 2
        self.batch_size = OPENAI_RERANK_EMBED_BATCH
        self.max_length = 512

    def inference(self, serialized_inputs):
        # 准备输入数据：从serialized_inputs中提取文本
        texts = [data for input_name, data in serialized_inputs.items()]
        # 调用OpenAI Embedding API获取嵌入向量
        start_time = time.time()
        embeddings_response = gpt_client.embeddings.create(
            input=texts,
            model=self.model_name)
        print('OpenAI Embedding API call time: {} s'.format(time.time() - start_time))

        # 从响应中提取嵌入向量，并转换为NumPy数组
        embeddings = np.array([item['embedding'] for item in embeddings_response['data']])

        # 模拟Triton输出格式：通常是一个具有特定名称的NumPy数组
        # 这里我们简化处理，直接返回NumPy数组，实际使用时可能需要进一步封装以匹配特定的输出结构
        return embeddings

    def merge_inputs(self, query_input, passage_input):
        # 由于不再基于token操作，该方法可用于模拟兼容的结构或直接处理文本
        # 为了简化，这里我们返回文本列表，实际不进行合并
        # 注意：实际使用时，应根据具体逻辑调整
        return [query_input, passage_input]

    def tokenize_preproc(self, query: str, passages: List[str]):
        # 模拟分词处理的结果，实际上将文本封装为预期的格式
        # 这里我们直接返回文本，实际不进行分词
        texts = []
        for passage in passages:
            merged_input = self.merge_inputs(query, passage)
            # 模拟返回结构，实际上只包含文本
            texts.append({'input_ids': merged_input, 'attention_mask': [0], 'token_type_ids': [0]})
        return texts, list(range(len(passages)))

    def predict(self, query: str, passages: List[str]):
        # 准备文本列表，包括查询和所有段落
        texts = [query] + passages
        # 调用inference方法获取嵌入向量
        embeddings = self.inference(texts)
        # 计算查询嵌入向量与每个段落嵌入向量之间的相似度
        query_embedding = embeddings[0]
        passage_embeddings = embeddings[1:]
        scores = [1 - cosine(query_embedding, passage_embedding) for passage_embedding in passage_embeddings]

        # 打印并返回得分
        print("Scores:", scores)
        return scores
