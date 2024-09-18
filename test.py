from tinygraph.graph import TinyGraph

graph = TinyGraph(uri="bolt://localhost:7687", user="neo4j", password="Fangshiyi0")
graph.load_document("mlbook/01_Introduction.md")