import re


def get_text_inside_tag(html_string: str, tag: str):
    # html_string 为待解析文本，tag为查找标签
    pattern = f"<{tag}>(.*?)<\/{tag}>"
    try:
        result = re.findall(pattern, html_string, re.DOTALL)
        return result
    except SyntaxError as e:
        raise ("Json Decode Error: {error}".format(error=e))


def save_triplets_to_txt(triplets, file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{triplets[0]},{triplets[1]},{triplets[2]}\n")


# def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
#         """
#         calculate cosine similarity between two vectors
#         """
#         dot_product = np.dot(vector1, vector2)
#         magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
#         if not magnitude:
#             return 0
#         return dot_product / magnitude

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
