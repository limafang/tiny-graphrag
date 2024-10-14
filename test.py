from neo4j import GraphDatabase
from tinygraph.graph import TinyGraph
from tinygraph.embedding.zhipu import zhipuEmb
from tinygraph.llm.zhipu import zhipuLLM


emb = zhipuEmb("")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))
llm = zhipuLLM("glm-4-plus", "")
graph = TinyGraph(driver=driver, llm=llm, emb=emb)

graph.load_document("/home/panduyi/edu/tiny-graphrag/mlbook/01_Introduction.md")
# graph.detect_communities()
# filtered_item = next((v for v in graph.community_schema().values() if v['sub_communities']), None)
filtered_item = graph.community_schema()["133"]
# 输出结果
if filtered_item:
    print(filtered_item)
else:
    print("No item with non-empty sub_communities found.")
print(graph.loaded_documents)
