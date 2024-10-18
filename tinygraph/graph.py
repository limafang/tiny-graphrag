from neo4j import GraphDatabase
import os
from tqdm import tqdm
from .utils import get_text_inside_tag
from .prompt import GET_ENTITY, GET_TRIPLETS, GEN_COMMUNITY_REPORT
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import json


class TinyGraph:

    def __init__(self, driver, llm, emb, working_dir="working_dir"):
        self.driver = driver
        self.llm = llm
        self.embedding = emb
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.txt_path = os.path.join(working_dir, "file.txt")
        self.chunk_path = os.path.join(working_dir, "chunk.json")
        self.community_path = os.path.join(working_dir, "community.json")

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
        if filepath in self.get_loaded_documents():
            print(f"Doc '{filepath}' has been loaded, skip import process.")
            return
        text_segments = self.split_text(filepath)
        print(f"Doc '{filepath}' has been loaded, processing.")
        for segment in tqdm(text_segments, desc=f"processing '{filepath}'"):
            entities = self.get_entity(segment)
            triplets = self.get_triplets(segment, entities)
            for subject, predicate, obj in triplets:
                self.create_triplet(subject, predicate, obj)
        print(f"doc '{filepath}' has been loaded.")

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

    def get_entity_by_name(self, name):
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            entities = [record["n"] for record in result]
        return entities[0]

    def get_edges_by_id(self, name):
        # TODO
        query = """
        MATCH (n:Entity {name: $name})-[r]-(m:Entity)
        RETURN r
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            edges = [record["r"] for record in result]
        return edges

    def gen_community_schema(self) -> dict[str, dict]:
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
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    def gen_community_schema(self):
        community_schema = self.gen_community_schema()
        with open(self.community_path, "w", encoding="utf-8") as file:
            json.dump(community_schema, file, indent=4)

    def read_community_schema(self):
        try:
            with open(self.community_path, "r", encoding="utf-8") as file:
                community_schema = json.load(file)
        except:
            raise FileNotFoundError(
                "Community schema not found. Please make sure to generate it first."
            )
        return community_schema

    def get_loaded_documents(self):
        try:
            with open(self.txt_path, "w", encoding="utf-8") as file:
                lines = file.readlines()
                return set(line.strip() for line in lines)
        except:
            raise FileNotFoundError("Cache file not found.")

    def add_loaded_documents(self, file_path):
        if file_path in self.loaded_documents:
            print(
                f"Document '{file_path}' has already been loaded, skipping addition to cache."
            )
            return
        with open(self.txt_path, "a", encoding="utf-8") as file:
            file.write(file_path + "\n")
        self.loaded_documents.add(file_path)

    def gen_single_community_report(self, community: dict):
        nodes = community["nodes"]
        edges = community["edges"]
        nodes_describe = [self.get_entity_by_name(i) for i in nodes]
        edges_describe = [self.get_entity_by_name(i) for i in edges]
        data = f"""
        Text:
        -----Entities-----
        ```csv
        {nodes_describe}
        ```
        -----Relationships-----
        ```csv
        {edges_describe}
        ```"""
        prompt = GEN_COMMUNITY_REPORT.format(input_text=data)
        report = self.llm.predict(prompt)
        return report

    def generate_community_report(self):
        communities_schema = self.read_community_schema()
        # 从level较小的社区开始生成报告
        communities_schema = dict(
            sorted(
                communities_schema.items(),
                key=lambda item: item[1]["level"],
                reverse=True,
            )
        )
        community_keys, community_values = list(communities_schema.keys()), list(
            communities_schema.values()
        )
        for i in tqdm(range(len(community_keys)), desc="generating community report"):
            community = community_values[i]
            community["report"] = self.gen_single_community_report(community)
            break
        # TODO save new community schema
