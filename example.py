from neo4j import GraphDatabase
from tinygraph.graph import TinyGraph
from tinygraph.embedding.zhipu import zhipuEmb
from tinygraph.llm.zhipu import zhipuLLM
from tinygraph.utils import compute_mdhash_id
from tinygraph.prompt import GET_TRIPLETS

emb = zhipuEmb("7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL", "embedding-3")
llm = zhipuLLM("glm-4-flash", "7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL")
graph = TinyGraph(driver=driver, llm=llm, emb=emb)
from neo4j import GraphDatabase
from tinygraph.graph import TinyGraph
from tinygraph.embedding.zhipu import zhipuEmb
from tinygraph.llm.zhipu import zhipuLLM
from tinygraph.utils import compute_mdhash_id
from tinygraph.prompt import GET_TRIPLETS

emb = zhipuEmb("7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL", "embedding-3")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "hellograph"))
llm = zhipuLLM("glm-4-flash", "7c172280522ff372cba10aeb1c67cbd2.6YOBziXCNFnAx1hL")
graph = TinyGraph(driver=driver, llm=llm, emb=emb)
