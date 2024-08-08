from groq import Chatbot
from prompt import *
from utils import get_text_inside_tag


def split_text(file_path, segment_length=300, overlap_length=50):
    """
    从文件中读取文字，并按照指定的长度划分文本，同时每个片段之间有指定的重叠区域。

    :param file_path: 要读取的文件的路径。
    :param segment_length: 每个片段的基准长度（不包括重叠部分），默认为400。
    :param overlap_length: 相邻片段之间的重叠长度，默认为100。
    :return: 包含文本片段的列表，每个片段都有指定的重叠区域（除了最后一个片段）。
    """
    if overlap_length >= segment_length:
        raise ValueError(
            "Overlap length cannot be greater than or equal to segment length."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    text_segments = []

    if len(content) >= segment_length:
        text_segments.append(content[:segment_length])

    start_index = segment_length - overlap_length
    while start_index + segment_length <= len(content):
        text_segments.append(content[start_index : start_index + segment_length])
        start_index += segment_length - overlap_length
    if start_index < len(content):
        text_segments.append(content[start_index:])

    return text_segments


def get_entity(text: str):
    llm = Chatbot()
    data = llm.predict(GET_ENTITY.format(text=text))
    data = get_text_inside_tag(data, "concept")
    return data


def get_triplets(text: str, entity: list):
    llm = Chatbot()
    data = llm.predict(GET_TRIPLETS.format(text=text, entity=entity))
    data = get_text_inside_tag(data, "triplet")
    res = []
    for i in data:
        try:
            subject = get_text_inside_tag(i, "subject")[0]
            predicate = get_text_inside_tag(i, "predicate")[0]
            object = get_text_inside_tag(i, "object")[0]
            res.append((subject, predicate, object))
        except:
            continue
    return res


from neo4j import GraphDatabase


class Neo4jHandler:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_triplet(self, subject, predicate, obj):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_and_return_triplet, subject, predicate, obj
            )

    @staticmethod
    def _create_and_return_triplet(tx, subject, predicate, obj):
        query = (
            "MERGE (a:Entity {name: $subject}) "
            "MERGE (b:Entity {name: $object}) "
            "MERGE (a)-[r:Relationship {name: $predicate}]->(b)"
            "RETURN a, b, r"
        )
        result = tx.run(query, subject=subject, object=obj, predicate=predicate)
        try:
            return [
                {"a": record["a"]["name"], "b": record["b"]["name"], "r": predicate}
                for record in result
            ]
        except Exception as e:
            print("An error occurred:", e)
            return None

    def query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        return records
