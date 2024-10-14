from sentence_transformers import SentenceTransformer
from typing import List
from base_emb import BaseEmb


class HfEmb(BaseEmb):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = SentenceTransformer(model_name)

    def get_emb(self, text: str) -> List[float]:
        emb = self.client.encode(text)
        return emb.tolist()
