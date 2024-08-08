from py2neo import Graph
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL")
# 请替换以下信息为你的Neo4j数据库的实际连接信息
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Fangshiyi0"



def get_emb(text):
    emb = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
    print(emb.data[0].embedding)
    return emb.data[0].embedding


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    calculate cosine similarity between two vectors
    """
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if not magnitude:
        return 0
    return dot_product / magnitude


graph = Graph(uri, auth=(username, password))

query = f"""
       MATCH (n) RETURN n.name as name
       """

nodes = graph.run(query).data()

for node in nodes:
    print(node)
    embedding = get_emb(node["name"])
    print(embedding)
    add_emb_query = f"""
           MATCH (n {{name: '{node['name']}'}})
           SET n.embedding = {embedding}
           """
    graph.run(add_emb_query)
