from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from zhipuai import ZhipuAI
from graph import Neo4jHandler

client = ZhipuAI(api_key="7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL")
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Fangshiyi0"
graph = Neo4jHandler(uri, username, password)


def get_emb(text):
    emb = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
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


user_input = "Hello! I do not know how to build a decision tree. Can you help me?"

input_emb = get_emb(user_input)

query = """
MATCH (n)
RETURN n.name, n.embedding
"""

res = []

nodes = graph.query(query)

for node in nodes:
    similarity = cosine_similarity(input_emb, node["n.embedding"])
    res.append((node["n.name"], similarity))

res_sorted = sorted(res, key=lambda x: x[1], reverse=True)

print(res_sorted[0][0])

info_query = f"""
MATCH (p)<-[r]-(q)
WHERE p.name = 'Decision Tree'
RETURN q.name,q.student_mastery
UNION
MATCH (p)-[r]->(q)
WHERE p.name = 'Decision Tree'
RETURN q.name,q.student_mastery;
"""

nodes = graph.query(info_query)

from groq import Chatbot
from prompt import TEST_PROMPT

teacher = Chatbot(system_prompt=TEST_PROMPT.format(state=nodes))

# teacher = Chatbot(system_prompt=TEST_PROMPT)

user_input = "Hello! I do not know how to build a decision tree. Can you help me?"

while True:
    response = teacher.send_message(user_input)
    print(response)
    user_input = input()
    print(user_input)
