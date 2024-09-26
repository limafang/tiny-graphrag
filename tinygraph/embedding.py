from zhipuai import ZhipuAI
from typing import List
from sentence_transformers import SentenceTransformer

class zhipuemb:
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)

    def get_emb(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model="embedding-3",
            input=text,
        )
        return emb.data[0].embedding

class Hf_Emb:
    def __init__(self):
        self.client = SentenceTransformer("embedding/models--Alibaba-NLP--gte-large-en-v1.5")
    def get_emb(self, text:str) -> List[float]:
        emb = self.client.encode(text)
        return emb.tolist()