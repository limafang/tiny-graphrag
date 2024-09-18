from neo4j import GraphDatabase
from .utils import get_text_inside_tag
from .prompt import GET_ENTITY, GET_TRIPLETS
from .llm import groq
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .embedding import zhipuemb

class TinyGraph:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm = groq()
        self.embedding = zhipuemb()
        self.loaded_documents = self._get_loaded_documents_from_db()

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


    def get_entity(self, text: str):
        data = self.llm.predict(GET_ENTITY.format(text=text))
        data = get_text_inside_tag(data, "concept")
        return data


    def get_triplets(self, text: str, entity: list):
        data = self.llm.predict(GET_TRIPLETS.format(text=text, entity=entity))
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
    
    
    def load_document(self, filepath):
        # 检查文档是否已经被加载
        if filepath in self.loaded_documents:
            print(f"doc '{filepath}' has been loaded, skip import process.")
            return
        text_segments = self.split_text(filepath)
        for segment in text_segments:
            entities = self.get_entity(segment)
            triplets = self.get_triplets(segment, entities)
            for subject, predicate, obj in triplets:
                self.create_triplet(subject, predicate, obj)
        self._add_document_to_db(filepath)
        self.loaded_documents.add(filepath)
        print(f"doc '{filepath}' has been loaded.")
        
        # 检测社区并创建社区节点
        self.detect_and_create_communities()

    def get_loaded_documents(self):
        """
        返回已加载的文档列表。
        """
        return self.loaded_documents

    def _add_document_to_db(self, filepath):
        """
        在图数据库中添加文档节点。
        """
        with self.driver.session() as session:
            session.write_transaction(self._create_document_node, filepath)

    @staticmethod
    def _create_document_node(tx, filepath):
        query = (
            "MERGE (d:Document {path: $filepath}) "
            "RETURN d"
        )
        tx.run(query, filepath=filepath)

    def _get_loaded_documents_from_db(self):
        """
        从图数据库中获取已加载的文档列表。
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_document_nodes)
        return set(result)

    @staticmethod
    def _get_document_nodes(tx):
        query = "MATCH (d:Document) RETURN d.path"
        result = tx.run(query)
        return [record["d.path"] for record in result]

    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    
    def get_embedding(self, text: str) -> List[float]:
        """
        get embedding of a text
        """
        return self.embedding.get_emb(text)
    
    def get_similar_nodes(self, input_emb: List[float]) -> List[Tuple[str, float]]:
        """
        根据输入的嵌入向量，返回与图数据库中节点的相似度排序列表。
        """
        query = """
        MATCH (n)
        RETURN n.name, n.embedding
        """
        nodes = self.query(query)
        res = []
        for node in nodes:
            similarity = self.cosine_similarity(input_emb, node["n.embedding"])
            res.append((node["n.name"], similarity))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def detect_communities(self):
        """
        使用 Leiden 算法检测图中的社区。
        """
        query = """
        CALL gds.leiden.write({
            nodeProjection: 'Entity',
            relationshipProjection: {
                Relationship: {
                    type: 'Relationship',
                    orientation: 'UNDIRECTED'
                }
            },
            writeProperty: 'community',
            maxIterations: 10,
            tolerance: 0.0001
        })
        YIELD communityCount, modularity, modularities
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                print(f"社区数量: {record['communityCount']}, 模块度: {record['modularity']}")

    def get_communities(self):
        """
        获取每个节点所属的社区。
        """
        query = """
        MATCH (n:Entity)
        RETURN n.name AS name, n.community AS community
        """
        with self.driver.session() as session:
            result = session.run(query)
            communities = {}
            for record in result:
                name = record["name"]
                community = record["community"]
                if community not in communities:
                    communities[community] = []
                communities[community].append(name)
        return communities

    def detect_and_create_communities(self):
        """
        使用 Leiden 算法检测社区，并创建社区节点。
        """
        self.detect_communities()
        communities = self.get_communities()
        for community, nodes in communities.items():
            community_size = len(nodes)
            level = self._determine_community_level(community_size)
            report = " ".join(nodes)
            self._create_community_node(community, level, report)

    def _determine_community_level(self, size):
        """
        根据社区大小确定社区等级。
        """
        if size < 10:
            return 1
        elif size < 50:
            return 2
        elif size < 100:
            return 3
        else:
            return 4

    def _create_community_node(self, community, level, report):
        """
        创建社区节点。
        """
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_community, community, level, report)

    @staticmethod
    def _create_and_return_community(tx, community, level, report):
        query = (
            "MERGE (c:Community {id: $community}) "
            "SET c.level = $level, c.report = $report "
            "RETURN c"
        )
        tx.run(query, community=community, level=level, report=report)