"""aiotcvectordb.model

对外暴露：
- 异步模型：AsyncDatabase / AsyncAIDatabase / AsyncCollection / AsyncCollectionView / AsyncDocumentSet
- 同步模型（仅类型定义，来自 vendor）：Document / Filter / AnnSearch / KeywordSearch / Rerank
- 索引与枚举（来自 vendor）：Index / IndexField / VectorIndex / FilterIndex / SparseIndex / SparseVector
  以及 FieldType / IndexType / MetricType / ReadConsistency
- Embedding 区分：
  - CollectionEmbedding: tcvectordb.model.collection.Embedding
  - ViewEmbedding      : tcvectordb.model.collection_view.Embedding
  - SplitterProcess / ParsingProcess 用于 CollectionView
"""

# 异步模型（本库实现）
from .database import AsyncDatabase
from .ai_database import AsyncAIDatabase
from .collection import AsyncCollection
from .collection_view import AsyncCollectionView
from .document_set import AsyncDocumentSet

# 同步模型与类型（从 vendor 透出，便于闭环）
from tcvectordb.model.document import (
    Document,
    Filter,
    AnnSearch,
    KeywordSearch,
    Rerank,
)
from tcvectordb.model.index import (
    Index,
    IndexField,
    VectorIndex,
    FilterIndex,
    SparseIndex,
    SparseVector,
)
from tcvectordb.model.enum import (
    FieldType,
    IndexType,
    MetricType,
    ReadConsistency,
)

# Embedding 与 CV 相关参数（为避免歧义，提供带前缀的别名）
from tcvectordb.model.collection import Embedding as CollectionEmbedding
from tcvectordb.model.collection_view import (
    Embedding as ViewEmbedding,
    SplitterProcess,
    ParsingProcess,
)

__all__ = [
    # async models
    "AsyncDatabase",
    "AsyncAIDatabase",
    "AsyncCollection",
    "AsyncCollectionView",
    "AsyncDocumentSet",
    # vendor document/types
    "Document",
    "Filter",
    "AnnSearch",
    "KeywordSearch",
    "Rerank",
    # index
    "Index",
    "IndexField",
    "VectorIndex",
    "FilterIndex",
    "SparseIndex",
    "SparseVector",
    # enums
    "FieldType",
    "IndexType",
    "MetricType",
    "ReadConsistency",
    # embeddings and processes
    "CollectionEmbedding",
    "ViewEmbedding",
    "SplitterProcess",
    "ParsingProcess",
]
