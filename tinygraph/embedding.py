from zhipuai import ZhipuAI
from typing import List
class zhipuemb:
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)

    def get_emb(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model="embedding-3",
            input=text,
        )
        return emb.data[0].embedding