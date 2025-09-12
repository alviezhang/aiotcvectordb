# aiotcvectordb — 腾讯云 VectorDB Python 异步 SDK

[![CI](https://github.com/alviezhang/aiotcvectordb/actions/workflows/ci.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/ci.yml)
[![TestPyPI](https://github.com/alviezhang/aiotcvectordb/actions/workflows/testpypi.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/testpypi.yml)
[![Publish](https://github.com/alviezhang/aiotcvectordb/actions/workflows/release.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/release.yml)

基于 `aiohttp` 的异步版腾讯云 VectorDB SDK，接口与官方 `tcvectordb` 同步版保持一致或等价，提供非阻塞 API 与更友好的 REPL 展示。

## 特性

- 纯异步 HTTP 客户端与连接池，支持代理。
- 从 `aiotcvectordb.model` 透出与 `tcvectordb` 一致的类型（索引、枚举、文档类型等）。
- 提供 `AsyncDatabase` / `AsyncCollection` / `AsyncCollectionView` / `AsyncDocumentSet` 异步封装。
- 支持向量检索、混合检索（稠密/稀疏）、服务端文本检索（自动 embedding）与全文检索。

## 环境要求

- Python 3.9+
- 依赖：`tcvectordb`、`aiohttp`、`numpy`
- 可选：`qcloud_cos`（`CollectionView.upload/load_and_split_text` 需要）

## 安装

```bash
pip install aiotcvectordb
```

源码安装（仓库根目录）：

```bash
pip install -e .
# 或使用 uv
uv pip install -e .
```

## 快速开始

```python
import asyncio
from aiotcvectordb import AsyncVectorDBClient
from aiotcvectordb.model import (
    Index, VectorIndex, FilterIndex,
    FieldType, IndexType, MetricType,
)

async def main():
    async with AsyncVectorDBClient(
        url="http://127.0.0.1:8081",
        username="root",
        key="<your-api-key>",
    ) as client:
        # 创建数据库（若不存在）
        await client.create_database_if_not_exists("demo_db")

        # 定义集合索引
        idx = Index()
        idx.add(VectorIndex(
            name="vector",
            dimension=3,
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params={"M": 8, "efConstruction": 80},
        ))
        idx.add(FilterIndex(
            name="id",
            field_type=FieldType.String,
            index_type=IndexType.PRIMARY_KEY,
        ))

        # 创建集合（若不存在）
        await client.create_collection_if_not_exists(
            database_name="demo_db",
            collection_name="demo_coll",
            shard=1,
            replicas=1,
            index=idx,
        )

        # 写入文档
        docs = [
            {"id": "1", "vector": [0.1, 0.2, 0.3], "tag": "hello"},
            {"id": "2", "vector": [0.2, 0.3, 0.1], "tag": "world"},
        ]
        await client.upsert("demo_db", "demo_coll", documents=docs)

        # 向量相似度检索
        res = await client.search(
            database_name="demo_db",
            collection_name="demo_coll",
            vectors=[[0.1, 0.2, 0.3]],
            limit=5,
            retrieve_vector=False,
        )
        print(res)

asyncio.run(main())
```

## 常用操作

- 数据库：`create_database`、`create_database_if_not_exists`、`drop_database`、`list_databases`
- 集合：`create_collection`、`create_collection_if_not_exists`、`describe_collection`、`list_collections`、`truncate_collection`、`set_alias`、`delete_alias`
- 文档：`upsert`、`query`、`count`、`update`、`delete`
- 检索：`search`、`search_by_id`、`search_by_text`（服务端 embedding）、`hybrid_search`、`fulltext_search`

## AI 文档库

```python
from aiotcvectordb.model import SplitterProcess

aidb = await client.create_ai_database("ai_demo")
cv = await aidb.create_collection_view(name="cv1")
ds = await cv.load_and_split_text("./doc.pdf", splitter_process=SplitterProcess())
results = await cv.search("你的问题", limit=5)
```

## 链接

- 仓库：https://github.com/alviezhang/aiotcvectordb
- 问题反馈：https://github.com/alviezhang/aiotcvectordb/issues

## 许可协议

MIT

