# Tiny-Graphrag

Tiny-Graphrag 是一个简洁版本的 GraphRAG 实现，旨在提供一个最简单的 GraphRAG 系统，包含所有必要的功能。我们实现了添加文档的全部流程，以及本地查询和全局查询的功能。

## 安装

Tiny-Graphrag 需要以下版本的 Neo4j 和 JDK，以及 GDS 插件：

- Neo4j: 5.24.0
- OpenJDK: 17.0.12
- GDS: 2.10.1

## 快速开始

首先克隆仓库：

```shell
git clone https://github.com/limafang/tiny-graphrag.git
cd tiny-graphrag
```

安装必要依赖：

```shell
pip install -r requirements.txt
```

接下来，你需要配置使用的 LLM 和 Embedding 服务。目前我们只支持 zhipu 的 LLM 和 Embedding 服务：

```python
from tinygraph.graph import TinyGraph
from tinygraph.embedding.zhipu import zhipuEmb
from tinygraph.llm.zhipu import zhipuLLM

emb = zhipuEmb("model name", "your key")
llm = zhipuLLM("model name", "your key")
graph = TinyGraph(
    url="your url",
    username="neo4j name",
    password="neo4j password",
    llm=llm,
    emb=emb,
)
```

使用 TinyGraph 添加文档。目前支持所有文本格式的文件。这一步的时间可能较长，结束后，在当前目录下会生成一个 `workspace` 文件夹，包含 `community`、`chunk` 和 `doc` 信息：

```python
graph.add_document("example/data.md")
```

完成文档添加后，可以使用 TinyGraph 进行查询。TinyGraph 支持本地查询和全局查询：

```python
local_res = graph.local_query("what is ML")
print(local_res)
global_res = graph.global_query("what is ML")
print(global_res)
```

通过以上步骤，你可以快速上手 Tiny-Graphrag，体验其强大的文档管理和查询功能。

## 代码解读
本仓库提供了Tiny-Graphrag项目核心代码的解读文档，用于帮助新手快速理解整个项目，详情见：Tiny-Graphrag 使用指南与代码解读.md

## 致谢

编写 Tiny-Graphrag 的过程中，我们参考了以下项目：

[GraphRAG](https://github.com/microsoft/graphrag)

[nano-graphrag](https://github.com/gusye1234/nano-graphrag)

需要说明的是，Tiny-Graphrag 是一个简化版本的 GraphRAG 实现，并不适用于生产环境，如果你需要一个更完整的 GraphRAG 实现，我们建议你使用上述项目。
