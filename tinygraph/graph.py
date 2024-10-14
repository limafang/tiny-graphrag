from neo4j import GraphDatabase
import os
from tqdm import tqdm
from .utils import get_text_inside_tag
from .prompt import GET_ENTITY, GET_TRIPLETS
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import json


class TinyGraph:

    def __init__(self, driver, llm, emb, cache_path="data_info.txt"):
        self.driver = driver
        self.llm = llm
        self.embedding = emb
        self.cache_path = cache_path
        self.loaded_documents = self.get_loaded_documents()

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

    @staticmethod
    def split_text(file_path, segment_length=300, overlap_length=50):
        if overlap_length >= segment_length:
            raise ValueError(
                "Overlap length cannot be greater than or equal to segment length."
            )

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        text_segments = []
        start_index = 0

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

        if filepath in self.loaded_documents:
            print(f"Doc '{filepath}' has been loaded, skip import process.")
            return
        text_segments = self.split_text(filepath)
        print(f"Doc '{filepath}' has been loaded, processing.")
        for segment in tqdm(text_segments, desc=f"processing '{filepath}'"):
            entities = self.get_entity(segment)
            triplets = self.get_triplets(segment, entities)
            for subject, predicate, obj in triplets:
                self.create_triplet(subject, predicate, obj)
        self.loaded_documents.add(filepath)
        print(f"doc '{filepath}' has been loaded.")

        # 检测社区并创建社区节点
        # self.detect_and_create_communities()

    def get_loaded_documents(self):
        if not os.path.exists(self.cache_path):
            with open(self.cache_path, "w", encoding="utf-8") as file:
                pass
        with open(self.cache_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return set(line.strip() for line in lines)

    def add_loaded_documents(self, file_path):
        if file_path in self.loaded_documents:
            print(
                f"Document '{file_path}' has already been loaded, skipping addition to cache."
            )
            return
        with open(self.cache_path, "a", encoding="utf-8") as file:
            file.write(file_path + "\n")
        self.loaded_documents.add(file_path)

    # def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
    #     """
    #     calculate cosine similarity between two vectors
    #     """
    #     dot_product = np.dot(vector1, vector2)
    #     magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    #     if not magnitude:
    #         return 0
    #     return dot_product / magnitude

    # def get_embedding(self, text: str) -> List[float]:
    #     """
    #     get embedding of a text
    #     """
    #     return self.embedding.get_emb(text)

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

    def detect_communities(self):
        query = """
        CALL gds.graph.project(
            'graph_help',
            ['Entity'],
            {
                Relationship: {
                    orientation: 'UNDIRECTED'
                }
            }
        )
        """
        with self.driver.session() as session:
            result = session.run(query)
            print(result)

        query = """
        CALL gds.leiden.write('graph_help', {
            writeProperty: 'communityIds',
            includeIntermediateCommunities: True,
            maxLevels: 10,
            tolerance: 0.0001,
            gamma: 1.0,
            theta: 0.01
        })
        YIELD communityCount, modularity, modularities
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                print(
                    f"社区数量: {record['communityCount']}, 模块度: {record['modularity']}"
                )
            session.run("CALL gds.graph.drop('graph_help')")

    def community_schema(self) -> dict[str, dict]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )

        with self.driver.session() as session:
            # Fetch community data
            result = session.run(
                f"""
                MATCH (n:Entity)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:Entity) | m.name] AS connected_nodes
                RETURN n.name AS node_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            max_num_ids = 0
            for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
            #         chunk_ids = source_id.split(GRAPH_FIELD_SEP)
            #         results[cluster_key]["chunk_ids"].update(chunk_ids)
            #         max_num_ids = max(
            #             max_num_ids, len(results[cluster_key]["chunk_ids"])
            #         )

            # # Process results
            # for k, v in results.items():
            #     v["edges"] = [list(e) for e in v["edges"]]
            #     v["nodes"] = list(v["nodes"])
            #     v["chunk_ids"] = list(v["chunk_ids"])
            #     v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    # def get_communities(self):
    #     """
    #     获取每个节点所属的社区。
    #     """
    #     query = """
    #     MATCH (n:Entity)
    #     RETURN id(n) AS nodeId,n.community AS community
    #     """
    #     with self.driver.session() as session:
    #         result = session.run(query)
    #         communities = {}
    #         for record in result:
    #             name = record["name"]
    #             community = record["community"]
    #             if community not in communities:
    #                 communities[community] = []
    #             communities[community].append(name)
    #     return communities

    # def detect_and_create_communities(self):
    #     """
    #     使用 Leiden 算法检测社区，并创建社区节点。
    #     """
    #     self.detect_communities()
    #     communities = self.get_communities()
    #     for community, nodes in communities.items():
    #         community_size = len(nodes)
    #         level = self._determine_community_level(community_size)
    #         report = " ".join(nodes)
    #         self._create_community_node(community, level, report)

    # def _determine_community_level(self, size):
    #     """
    #     根据社区大小确定社区等级。
    #     """
    #     if size < 10:
    #         return 1
    #     elif size < 50:
    #         return 2
    #     elif size < 100:
    #         return 3
    #     else:
    #         return 4

    # def _create_community_node(self, community, level, report):
    #     """
    #     创建社区节点。
    #     """
    #     with self.driver.session() as session:
    #         session.write_transaction(self._create_and_return_community, community, level, report)

    # @staticmethod
    # def _create_and_return_community(tx, community, level, report):
    #     query = (
    #         "MERGE (c:Community {id: $community}) "
    #         "SET c.level = $level, c.report = $report "
    #         "RETURN c"
    #     )
    #     tx.run(query, community=community, level=level, report=report)
