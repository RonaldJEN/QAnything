"""Wrapper around YouDao embedding models."""
from typing import List
from qanything_kernel.configs.model_config import OPENAI_EMBED_MODEL_NAME, OPENAI_EMBED_BATCH, OPENAI_EMBED_MODEL_VERSION
from qanything_kernel.models.gpt_loader import gpt_client
import concurrent.futures



class YouDaoLocalEmbeddings:
    def __init__(self):
        pass

    def _get_embedding(self, queries):
        embeddings = [gpt_client.embeddings.create(input=[j], model=OPENAI_EMBED_MODEL_NAME).data[0].embedding for j in queries]
        return embeddings

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = OPENAI_EMBED_BATCH
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            for future in futures:
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    @property
    def embed_version(self):
        return OPENAI_EMBED_MODEL_VERSION
