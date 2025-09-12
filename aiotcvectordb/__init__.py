"""
aiotcvectordb public API

Top-level只暴露客户端入口，模型与类型请从 aiotcvectordb.model 引入。
"""

from .client.stub import AsyncVectorDBClient

__all__ = [
    "AsyncVectorDBClient",
]
