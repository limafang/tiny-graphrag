from neo4j import GraphDatabase

# 创建一个连接到Neo4j数据库的驱动实例
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Fangshiyi0"  # 替换为你的Neo4j密码

# 创建驱动实例
driver = GraphDatabase.driver(uri, auth=(username, password))

info_query = f"""
MATCH (p)<-[r]-(q)
WHERE p.name = 'Decision Tree'
RETURN p.name, r.relation, q.name AS connected_node;
"""

# 使用一个session执行查询
with driver.session() as session:
    result = session.run(info_query)

    # 获取所有结果
    records = list(result)
    print(records)
    print(f"Number of results: {len(records)}")

# 关闭驱动，释放资源
driver.close()
