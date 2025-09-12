# aiotcvectordb — Tencent VectorDB Python Async SDK

[![CI](https://github.com/alviezhang/aiotcvectordb/actions/workflows/ci.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/ci.yml)
[![TestPyPI](https://github.com/alviezhang/aiotcvectordb/actions/workflows/testpypi.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/testpypi.yml)
[![Publish](https://github.com/alviezhang/aiotcvectordb/actions/workflows/release.yml/badge.svg)](https://github.com/alviezhang/aiotcvectordb/actions/workflows/release.yml)

An asyncio-first client for Tencent Cloud VectorDB built on top of `aiohttp`. It mirrors the official `tcvectordb` SDK’s models and request payloads, while providing non-blocking APIs and REPL-friendly representations.

Looking for Chinese docs? See README_zh.md.

## Features

- Fully async HTTP client using `aiohttp` with connection pooling and proxy support.
- Type parity with `tcvectordb` (indexes, enums, document types) re-exported under `aiotcvectordb.model`.
- Convenient async wrappers: `AsyncDatabase`, `AsyncCollection`, `AsyncCollectionView`, `AsyncDocumentSet`.
- Supports vector search, hybrid search (dense/sparse), text search with server-side embeddings, and full-text search.

## Requirements

- Python 3.9+
- Dependencies: `tcvectordb`, `aiohttp`, `numpy`
- Optional: `qcloud_cos` for AI document upload in `CollectionView.upload/load_and_split_text`

## Install

```bash
pip install aiotcvectordb
```

From source (repo root):

```bash
pip install -e .
# or with uv
uv pip install -e .
```

## Quickstart

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
        # Create database if not exists
        await client.create_database_if_not_exists("demo_db")

        # Define indexes for the collection
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

        # Create collection if not exists
        await client.create_collection_if_not_exists(
            database_name="demo_db",
            collection_name="demo_coll",
            shard=1,
            replicas=1,
            index=idx,
        )

        # Upsert documents
        docs = [
            {"id": "1", "vector": [0.1, 0.2, 0.3], "tag": "hello"},
            {"id": "2", "vector": [0.2, 0.3, 0.1], "tag": "world"},
        ]
        await client.upsert("demo_db", "demo_coll", documents=docs)

        # Vector similarity search
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

## Common Operations

- Databases: `create_database`, `create_database_if_not_exists`, `drop_database`, `list_databases`
- Collections: `create_collection`, `create_collection_if_not_exists`, `describe_collection`, `list_collections`, `truncate_collection`, `set_alias`, `delete_alias`
- Documents: `upsert`, `query`, `count`, `update`, `delete`
- Search: `search`, `search_by_id`, `search_by_text` (server-side embedding), `hybrid_search`, `fulltext_search`

## AI Document Database

```python
from aiotcvectordb.model import SplitterProcess

aidb = await client.create_ai_database("ai_demo")
cv = await aidb.create_collection_view(name="cv1")
ds = await cv.load_and_split_text("./doc.pdf", splitter_process=SplitterProcess())
results = await cv.search("your question", limit=5)
```

## Links

- Repo: https://github.com/alviezhang/aiotcvectordb
- Issues: https://github.com/alviezhang/aiotcvectordb/issues

## License

MIT
