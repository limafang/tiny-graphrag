import re
import numpy as np
from typing import List, Tuple
from hashlib import md5
import json


def get_text_inside_tag(html_string: str, tag: str):
    # html_string 为待解析文本，tag为查找标签
    pattern = f"<{tag}>(.*?)<\/{tag}>"
    try:
        result = re.findall(pattern, html_string, re.DOTALL)
        return result
    except SyntaxError as e:
        raise ("Json Decode Error: {error}".format(error=e))


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except:
        return {}


def write_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def save_triplets_to_txt(triplets, file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{triplets[0]},{triplets[1]},{triplets[2]}\n")


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    calculate cosine similarity between two vectors
    """
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if not magnitude:
        return 0
    return dot_product / magnitude


# def get_similar_nodes(self, input_emb: List[float]) -> List[Tuple[str, float]]:
#     """
#     根据输入的嵌入向量，返回与图数据库中节点的相似度排序列表。
#     """
#     query = """
#     MATCH (n)
#     RETURN n.name, n.embedding
#     """
#     nodes = self.query(query)
#     res = []
#     for node in nodes:
#         similarity = self.cosine_similarity(input_emb, node["n.embedding"])
#         res.append((node["n.name"], similarity))
#     return sorted(res, key=lambda x: x[1], reverse=True)
