from neo4j import GraphDatabase
import os
from tqdm import tqdm
from .utils import (
    get_text_inside_tag,
    cosine_similarity,
    compute_mdhash_id,
    read_json_file,
    write_json_file,
)
from .llm.base import BaseLLM
from .embedding.base import BaseEmb
from .prompt import (
    GET_ENTITY,
    GET_TRIPLETS,
    GEN_COMMUNITY_REPORT,
    ENTITY_DISAMBIGUATION,
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import json

from dataclasses import dataclass


@dataclass
class Entity:
    name: str
    description: str
    chunks_id: List[str]
    entity_id: str


@dataclass
class Triplet:
    subject: str
    subject_id: str
    predicate: str
    object: str
    object_id: str


class TinyGraph:

    def __init__(
        self, driver, llm: BaseLLM, emb: BaseLLM, working_dir: str = "workspace"
    ):
        self.driver = driver
        self.llm = llm
        self.embedding = emb
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.doc_path = os.path.join(working_dir, "doc.txt")
        self.chunk_path = os.path.join(working_dir, "chunk.json")
        self.community_path = os.path.join(working_dir, "community.json")
        self.loaded_documents = self.get_loaded_documents()

    def create_triplet(self, subject: dict, predicate, obj: dict):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_and_return_triplet, subject, predicate, obj
            )

    @staticmethod
    def _create_and_return_triplet(tx, subject, predicate, obj):
        query = (
            "MERGE (a:Entity {name: $subject_name, description: $subject_desc, chunks_id: $subject_chunks_id, entity_id: $subject_entity_id}) "
            "MERGE (b:Entity {name: $object_name, description: $object_desc, chunks_id: $object_chunks_id, entity_id: $object_entity_id}) "
            "MERGE (a)-[r:Relationship {name: $predicate}]->(b)"
            "RETURN a, b, r"
        )
        result = tx.run(
            query,
            subject_name=subject["name"],
            subject_desc=subject["description"],
            subject_chunks_id=subject["chunks id"],
            subject_entity_id=subject["entity id"],
            object_name=obj["name"],
            object_desc=obj["description"],
            object_chunks_id=obj["chunks id"],
            object_entity_id=obj["entity id"],
            predicate=predicate,
        )
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
        chunks = {}
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        text_segments = []
        start_index = 0

        while start_index + segment_length <= len(content):
            text_segments.append(content[start_index : start_index + segment_length])
            start_index += segment_length - overlap_length

        if start_index < len(content):
            text_segments.append(content[start_index:])

        for i in text_segments:
            chunks.update({compute_mdhash_id(i, prefix="chunk-"): i})

        return chunks

    def get_entity(self, text: str, chunk_id: str):
        data = self.llm.predict(GET_ENTITY.format(text=text))
        concepts = []
        for concept_html in get_text_inside_tag(data, "concept"):
            concept = {}
            concept["name"] = get_text_inside_tag(concept_html, "name")[0].strip()
            concept["description"] = get_text_inside_tag(concept_html, "description")[
                0
            ].strip()
            concept["chunks id"] = [chunk_id]
            concept["entity id"] = compute_mdhash_id(
                concept["description"], prefix="entity-"
            )
            concepts.append(concept)
        return concepts

    def get_triplets(self, content, entity: list) -> List[Dict]:
        data = self.llm.predict(GET_TRIPLETS.format(text=content, entity=entity))
        data = get_text_inside_tag(data, "triplet")
        res = []
        for i in data:
            try:
                subject = get_text_inside_tag(i, "subject")[0]
                subject_id = get_text_inside_tag(i, "subject_id")[0]
                predicate = get_text_inside_tag(i, "predicate")[0]
                object = get_text_inside_tag(i, "object")[0]
                object_id = get_text_inside_tag(i, "object_id")[0]
                res.append(
                    {
                        "subject": subject,
                        "subject_id": subject_id,
                        "predicate": predicate,
                        "object": object,
                        "object_id": object_id,
                    }
                )
            except:
                continue
        return res

    def add_document(self, filepath, use_llm_deambiguation=False):
        """
        Adds a document to the system by performing the following steps:
        1. Check if the document has already been loaded.
        2. Split the document into chunks.
        3. Extract entities and triplets from the chunks.
        4. Perform entity disambiguation if required.
        5. Merge entities and triplets.
        6. Store the merged entities and triplets in a Neo4j database.

        Args:
            filepath (str): The path to the document to be added.
            use_llm_deambiguation (bool): Whether to use LLM for entity disambiguation.
        """

        # ================ Check if the document has been loaded ================
        if filepath in self.get_loaded_documents():
            print(
                f"Document '{filepath}' has already been loaded, skipping import process."
            )
            return

        # ================ Chunking ================
        chunks = self.split_text(filepath)
        existing_chunks = read_json_file(self.chunk_path)

        # Filter out chunks that are already in storage
        new_chunks = {k: v for k, v in chunks.items() if k not in existing_chunks}

        if not new_chunks:
            print("All chunks are already in the storage.")
            return

        # Merge new chunks with existing chunks
        all_chunks = {**existing_chunks, **new_chunks}
        write_json_file(all_chunks, self.chunk_path)
        print(f"Document '{filepath}' has been chunked.")

        # ================ Entity Extraction ================
        all_entities = []
        all_triplets = []

        for chunk_id, chunk_content in tqdm(
            all_chunks.items(), desc=f"Processing '{filepath}'"
        ):
            try:
                entities = self.get_entity(chunk_content, chunk_id=chunk_id)
                all_entities.extend(entities)
                triplets = self.get_triplets(chunk_content, entities)
                all_triplets.extend(triplets)
            except:
                print(
                    f"An error occurred while processing chunk '{chunk_id}'. SKIPPING..."
                )

        print(
            f"{len(all_entities)} entities and {len(all_triplets)} triplets have been extracted."
        )
        # ================ Entity Disambiguation ================
        entity_names = list(set(entity["name"] for entity in all_entities))

        if use_llm_deambiguation:
            entity_id_mapping = {}
            for name in entity_names:
                same_name_entities = [
                    entity for entity in all_entities if entity["name"] == name
                ]
                transform_text = self.llm.predict(
                    ENTITY_DISAMBIGUATION.format(same_name_entities)
                )
                entity_id_mapping.update(
                    get_text_inside_tag(transform_text, "transform")
                )
        else:
            entity_id_mapping = {}
            for entity in all_entities:
                entity_name = entity["name"]
                if entity_name not in entity_id_mapping:
                    entity_id_mapping[entity_name] = entity["entity id"]

        # 根据 mapping 对所有实体进行消岐
        for entity in all_entities:
            entity["entity id"] = entity_id_mapping.get(
                entity["name"], entity["entity id"]
            )

        # 根据 mapping 对所有三元组进行消岐
        triplets_to_remove = [
            triplet
            for triplet in all_triplets
            if entity_id_mapping.get(triplet["subject"], triplet["subject_id"]) is None
            or entity_id_mapping.get(triplet["object"], triplet["object_id"]) is None
        ]

        updated_triplets = [
            {
                **triplet,
                "subject_id": entity_id_mapping.get(
                    triplet["subject"], triplet["subject_id"]
                ),
                "object_id": entity_id_mapping.get(
                    triplet["object"], triplet["object_id"]
                ),
            }
            for triplet in all_triplets
            if triplet not in triplets_to_remove
        ]

        all_triplets = updated_triplets

        unique_entity_ids = list(set(entity["entity id"] for entity in all_entities))

        # ================ Merge Entities ================
        merged_entities = []

        for entity_id in unique_entity_ids:
            same_id_entities = [
                entity for entity in all_entities if entity["entity id"] == entity_id
            ]
            description = " ".join(
                [entity["description"] for entity in same_id_entities]
            )
            chunk_ids = [entity["chunks id"] for entity in same_id_entities]

            merged_entities.append(
                {
                    "name": same_id_entities[0]["name"],
                    "description": description,
                    "chunks id": chunk_ids,
                    "entity id": entity_id,
                }
            )

        # ================ Store Data in Neo4j ================
        for triplet in all_triplets:
            subject_id = triplet["subject_id"]
            object_id = triplet["object_id"]
            subject = next(
                entity
                for entity in merged_entities
                if entity["entity id"] == subject_id
            )
            object = next(
                entity for entity in merged_entities if entity["entity id"] == object_id
            )
            self.create_triplet(subject, triplet["predicate"], object)
        # ================ communities ================
        # self.detect_communities()
        self.add_loaded_documents(filepath)
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
            entities = [record["n"].get("name") for record in result]
        return entities[0]

    def get_node_edgs(self, node):
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.name = $name
        RETURN n.name AS n,r.name AS r,m.name AS m
        """
        with self.driver.session() as session:
            result = session.run(query, name=node)
            edges = [(record["n"], record["r"], record["m"]) for record in result]
        # TODO need format
        return edges

    def get_node_chunks(self, node):
        # TODO
        pass

    def add_embedding_for_graph(self):
        query = """
        MATCH (n)
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                node = record["n"]
                name = node["name"]
                embedding = self.embedding.get_emb(name)
                # 更新节点，添加新的 embedding 属性
                update_query = """
                MATCH (n {name: $name})
                SET n.embedding = $embedding
                """
                session.run(update_query, name=name, embedding=embedding)

    def get_topk_similar_entities(self, input_emb, k=1) -> List[Tuple[str, float]]:
        query = """
        MATCH (n)
        RETURN n.name, n.embedding
        """
        nodes = self.query(query)
        res = []
        for node in nodes:
            if node["n.embedding"] is not None:
                similarity = cosine_similarity(input_emb, node["n.embedding"])
                res.append(node["n.name"])
        return sorted(res, key=lambda x: x[1], reverse=True)[:k]

    def get_communities(self, nodes: List):
        nodes = set(nodes)
        communities_schema = self.read_community_schema()
        res = []
        for community_id, community_info in communities_schema.items():
            if nodes & set(community_info["nodes"]):
                res.append(
                    {"community_id": community_id, "community_info": community_info}
                )
        return res

    def get_relations(self, nodes: List, input_emb):
        nodes = set(nodes)
        res = []
        for i in nodes:
            res.append(self.get_node_edgs(i))
        return res

    def get_chunks(self, nodes, input_emb):
        chunks = []
        for i in nodes:
            chunks.append(self.get_node_edgs(i))
        pass

    def get_edges_by_names(self, name1, name2):
        query = """
        MATCH (n:Entity {name: $name1})-[r]-(m:Entity {name: $name2})
        RETURN r
        """
        with self.driver.session() as session:
            result = session.run(query, {"name1": name1, "name2": name2})
            edges = [record["r"].get("name") for record in result]
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

    def gen_community(self):
        self.detect_communities()
        community_schema = self.gen_community_schema()
        with open(self.community_path, "w", encoding="utf-8") as file:
            json.dump(community_schema, file, indent=4)

    def read_community_schema(self) -> dict:
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
            with open(self.doc_path, "r", encoding="utf-8") as file:
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
        with open(self.doc_path, "a", encoding="utf-8") as file:
            file.write(file_path + "\n")
        self.loaded_documents.add(file_path)

    def gen_single_community_report(self, community: dict):
        nodes = community["nodes"]
        edges = community["edges"]
        nodes_describe = [
            {"name": i, "desc": self.get_entity_by_name(i)} for i in nodes
        ]
        edges_describe = [
            {
                "source": i[0],
                "target": i[1],
                "desc": self.get_edges_by_names(i[0], i[1]),
            }
            for i in edges
        ]
        nodes_csv = "entity,description\n"
        for node in nodes_describe:
            nodes_csv += f"{node['name']},{node['desc']}\n"
        edges_csv = "source,target,description\n"
        for edge in edges_describe:
            edges_csv += f"{edge['source']},{edge['target']},{edge['desc']}\n"
        data = f"""
        Text:
        -----Entities-----
        ```csv
        {nodes_csv}
        ```
        -----Relationships-----
        ```csv
        {edges_csv}
        ```"""
        prompt = GEN_COMMUNITY_REPORT.format(input_text=data)
        report = self.llm.predict(prompt)
        return report

    def generate_community_report(self):
        communities_schema = self.read_community_schema()
        # 从level较小的社区开始生成报告
        communities_schema = sorted(
            communities_schema.items(), key=lambda item: item[1]["level"], reverse=True
        )
        for community_key, community in tqdm(
            communities_schema, desc="generating community report"
        ):
            community["report"] = self.gen_single_community_report(community)
        with open(self.community_path, "w", encoding="utf-8") as file:
            json.dump(communities_schema, file, indent=4)
        print("All community report has been generated.")

    def build_local_query_context(self, query):
        topk_similar_entities_context = self.get_topk_similar_entities(query)
        topk_similar_communities_context = self.get_communities(
            topk_similar_entities_context, query
        )
        topk_similar_relations_context = self.get_relations(
            topk_similar_entities_context, query
        )
        topk_similar_chunks_context = self.get_chunks(
            topk_similar_entities_context, query
        )
        return f"""
        -----Reports-----
        ```csv
        {topk_similar_communities_context}
        ```
        -----Entities-----
        ```csv
        {topk_similar_entities_context}
        ```
        -----Relationships-----
        ```csv
        {topk_similar_relations_context}
        ```
        -----Sources-----
        ```csv
        {topk_similar_chunks_context}
        ```
        """

    def build_global_query_context(self):
        # TODO
        pass

    def local_query(self, query):
        context = self.build_local_query_context(query)
        prompt = f"""
        {context}
        """
        response = self.llm.predict(context)
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        return records

    def global_query(self, query):
        context = self.build_global_query_context()
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        return records
