from typing import List, Optional, Union, Dict, Any
from numpy import ndarray

from aiotcvectordb import exceptions
from aiotcvectordb.model.ai_database import AsyncAIDatabase
from aiotcvectordb.model.database import AsyncDatabase
from tcvectordb.model.collection import Embedding, FilterIndexConfig, Collection
from tcvectordb.model.document import Document, Filter, AnnSearch, KeywordSearch, Rerank
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import (
    FilterIndex,
    VectorIndex,
    Index,
    IndexField,
    SparseVector,
)

from aiotcvectordb.client.httpclient import AsyncHTTPClient


class AsyncVectorDBClient:
    """Async client for vector db using aiohttp."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: str = "",
        key: str = "",
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        timeout: int = 10,
        pool_size: int = 10,
        proxies: Optional[dict] = None,
        password: Optional[str] = None,
        connector: Optional[object] = None,
    ):
        self._conn = AsyncHTTPClient(
            url,
            username,
            key,
            timeout=timeout,
            pool_size=pool_size,
            proxies=proxies,
            password=password,
            connector=connector,
        )
        self._read_consistency = read_consistency

    @property
    def http_client(self):
        return self._conn

    async def close(self):
        await self._conn.close()

    async def __aenter__(self) -> "AsyncVectorDBClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def create_database(
        self, database_name: str, timeout: Optional[float] = None
    ) -> AsyncDatabase:
        """Creates a database.

        Args:
            database_name (str): The name of the database. A database name can only include
                numbers, letters, and underscores, and must not begin with a letter, and length
                must between 1 and 128
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            AsyncDatabase: A database object for async api.
        """
        db = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        await db.create_database(timeout=timeout)
        return db

    async def create_database_if_not_exists(
        self,
        database_name: str,
        timeout: Optional[float] = None,
    ) -> AsyncDatabase:
        """Create the database if it doesn't exist.

        Args:
            database_name (str): The name of the database. A database name can only include
                numbers, letters, and underscores, and must not begin with a letter, and length
                must between 1 and 128
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            AsyncDatabase: A database object.
        """
        dbs = await self.list_databases(timeout=timeout)
        for db in dbs:
            if db.database_name == database_name:
                return db
        return await self.create_database(database_name=database_name, timeout=timeout)

    async def create_ai_database(
        self, database_name: str, timeout: Optional[float] = None
    ) -> AsyncAIDatabase:
        """Creates an AI doc database.

        Args:
            database_name (str): The name of the database. A database name can only include
                numbers, letters, and underscores, and must not begin with a letter, and length
                must between 1 and 128
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            AIDatabase: A database object.
        """
        db = AsyncAIDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        await db.create_database(timeout=timeout)
        return db

    async def drop_database(
        self, database_name: str, timeout: Optional[float] = None
    ) -> Dict:
        """Delete a database.

        Args:
            database_name (str): The name of the database to delete.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            Dict: Contains code、msg、affectedCount
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.drop_database(timeout=timeout)

    async def drop_ai_database(
        self, database_name: str, timeout: Optional[float] = None
    ) -> Dict:
        """Delete an AI Database.

        Args:
            database_name (str): The name of the database to delete.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            Dict: Contains code、msg、affectedCount
        """
        res = await self._conn.post(
            "/ai/database/drop", {"database": database_name}, timeout
        )
        return res.data()

    async def list_databases(
        self, timeout: Optional[float] = None
    ) -> List[Union[AsyncDatabase, AsyncAIDatabase]]:
        """List all databases.

        Args:
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            List: all AsyncDatabase and AsyncAIDatabase
        """
        db = AsyncDatabase(conn=self._conn, read_consistency=self._read_consistency)
        dbs = await db.list_databases(timeout=timeout)
        return dbs

    async def database(self, database: str) -> Union[AsyncDatabase, AsyncAIDatabase]:
        """Get a database.

        Args:
            database (str): The name of the database.

        Returns:
            An AsyncDatabase or AsyncAIDatabase object
        """
        dbs = await self.list_databases()
        for db in dbs:
            if db.database_name == database:
                return db
        raise exceptions.ParamError(
            code=14100, message="Database not exist: {}".format(database)
        )

    async def create_collection(
        self,
        database_name: str,
        collection_name: str,
        shard: int,
        replicas: int,
        description: str = None,
        index: Index = None,
        embedding: Embedding = None,
        timeout: float = None,
        ttl_config: dict = None,
        filter_index_config: FilterIndexConfig = None,
        indexes: List[IndexField] = None,
    ) -> Collection:
        """Create a collection.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection. A collection name can only include
                numbers, letters, and underscores, and must not begin with a letter, and length
                must between 1 and 128
            shard (int): The shard number of the collection. Shard will divide a large dataset into smaller subsets.
            replicas (int): The replicas number of the collection. Replicas refers to the number of identical copies
                of each primary shard, used for disaster recovery and load balancing.
            description (str): An optional description of the collection.
            index (Index): A list of the index properties for the documents in a collection.
            embedding (Embedding): An optional embedding for embedding text when upsert documents.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.
            ttl_config (dict): TTL configuration, when set {'enable': True, 'timeField': 'expire_at'} means
                that ttl is enabled and automatically removed when the time set in the expire_at field expires
            filter_index_config (FilterIndexConfig): Enabling full indexing mode.
                Where all scalar fields are indexed by default.
            indexes (List[IndexField]): A list of the index properties for the documents in a collection.
        Returns:
            A Collection object.
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.create_collection(
            name=collection_name,
            shard=shard,
            replicas=replicas,
            description=description,
            index=index,
            embedding=embedding,
            timeout=timeout,
            ttl_config=ttl_config,
            filter_index_config=filter_index_config,
            indexes=indexes,
        )

    async def create_collection_if_not_exists(
        self,
        database_name: str,
        collection_name: str,
        shard: int,
        replicas: int,
        description: str = None,
        index: Index = None,
        embedding: Embedding = None,
        timeout: float = None,
        ttl_config: dict = None,
        filter_index_config: FilterIndexConfig = None,
        indexes: List[IndexField] = None,
    ) -> Collection:
        """Create the collection if it doesn't exist.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection. A collection name can only include
                numbers, letters, and underscores, and must not begin with a letter, and length
                must between 1 and 128
            shard (int): The shard number of the collection. Shard will divide a large dataset into smaller subsets.
            replicas (int): The replicas number of the collection. Replicas refers to the number of identical copies
                of each primary shard, used for disaster recovery and load balancing.
            description (str): An optional description of the collection.
            index (Index): A list of the index properties for the documents in a collection.
            embedding (``Embedding``): An optional embedding for embedding text when upsert documents.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.
            ttl_config (dict): TTL configuration, when set {'enable': True, 'timeField': 'expire_at'} means
                that ttl is enabled and automatically removed when the time set in the expire_at field expires
            filter_index_config (FilterIndexConfig): Enabling full indexing mode.
                Where all scalar fields are indexed by default.
            indexes (List[IndexField]): A list of the index properties for the documents in a collection.

        Returns:
            Collection: A collection object.
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.create_collection_if_not_exists(
            name=collection_name,
            shard=shard,
            replicas=replicas,
            description=description,
            index=index,
            embedding=embedding,
            timeout=timeout,
            ttl_config=ttl_config,
            filter_index_config=filter_index_config,
            indexes=indexes,
        )

    async def exists_collection(self, database_name: str, collection_name: str) -> bool:
        """Check if the collection exists.

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection to check.

        Returns:
            Bool: True if collection exists else False.
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.exists_collection(collection_name)

    async def describe_collection(
        self, database_name: str, collection_name: str, timeout: Optional[float] = None
    ) -> Collection:
        """Get a Collection by name.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            A Collection object.
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.describe_collection(collection_name, timeout=timeout)

    async def collection(self, database_name: str, collection_name: str) -> Collection:
        """Get a Collection by name.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.

        Returns:
            A Collection object
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.describe_collection(collection_name)

    async def list_collections(
        self, database_name: str, timeout: Optional[float] = None
    ) -> List[Collection]:
        """List all collections in the database.

        Args:
            database_name (str): The name of the database.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            List: all Collections
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.list_collections(timeout=timeout)

    async def drop_collection(
        self, database_name: str, collection_name: str, timeout: Optional[float] = None
    ) -> Dict:
        """Delete a collection by name.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            Dict: Contains code、msg、affectedCount
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.drop_collection(collection_name, timeout=timeout)

    async def truncate_collection(
        self, database_name: str, collection_name: str
    ) -> Dict:
        """Clear all the data and indexes in the Collection.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.

        Returns:
            Dict: Contains affectedCount
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.truncate_collection(collection_name)

    async def set_alias(
        self, database_name: str, collection_name: str, collection_alias: str
    ) -> Dict:
        """Set alias for collection.

        Args:
            database_name (str): The name of the database.
            collection_name  : The name of the collection.
            collection_alias : alias name to set.
        Returns:
            Dict: Contains affectedCount
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.set_alias(
            collection_name=collection_name, collection_alias=collection_alias
        )

    async def delete_alias(self, database_name: str, alias: str) -> Dict:
        """Delete alias by name.

        Args:
            database_name (str): The name of the database.
            alias  : alias name to delete.

        Returns:
            Dict: Contains affectedCount
        """
        adb = AsyncDatabase(
            conn=self._conn, name=database_name, read_consistency=self._read_consistency
        )
        return await adb.delete_alias(alias=alias)

    async def upsert(
        self,
        database_name: str,
        collection_name: str,
        documents: List[Union[Document, Dict]],
        timeout: Optional[float] = None,
        build_index: bool = True,
        **kwargs,
    ):
        """Upsert documents into a collection.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            documents (List[Union[Document, Dict]]) : The list of the document object or dict to upsert. Maximum 1000.
            timeout (float) : An optional duration of time in seconds to allow for the request.
                              When timeout is set to None, will use the connect timeout.
            build_index (bool) : An option for build index time when upsert, if build_index is true, will build index
                                 immediately, it will affect performance of upsert. And param buildIndex has same
                                 semantics with build_index, any of them false will be false

        Returns:
            Dict: Contains affectedCount
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.upsert(
            documents=documents, timeout=timeout, build_index=build_index, **kwargs
        )

    async def delete(
        self,
        database_name: str,
        collection_name: str,
        document_ids: List[str] = None,
        filter: Union[Filter, str] = None,
        timeout: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        """Delete document by conditions.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            document_ids (List[str]): The list of the document id
            filter (Union[Filter, str]): Filter condition of the scalar index field
            limit (int): The amount of document deleted, with a range of [1, 16384].
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.

        Returns:
            Dict: Contains affectedCount
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.delete(
            document_ids=document_ids, filter=filter, timeout=timeout, limit=limit
        )

    async def update(
        self,
        database_name: str,
        collection_name: str,
        data: Union[Document, Dict],
        filter: Union[Filter, str] = None,
        document_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Update document by conditions.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            data (Union[Document, Dict]): Set the fields to be updated.
            document_ids (List[str]): The list of the document id
            filter (Union[Filter, str]): Filter condition of the scalar index field
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.

        Returns:
            Dict: Contains affectedCount
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.update(
            data=data, filter=filter, document_ids=document_ids, timeout=timeout
        )

    async def query(
        self,
        database_name: str,
        collection_name: str,
        document_ids: Optional[List] = None,
        retrieve_vector: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filter: Union[Filter, str] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        sort: Optional[dict] = None,
    ) -> List[Dict]:
        """Query documents that satisfies the condition.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            document_ids (List[str]): The list of the document id
            retrieve_vector (bool): Whether to return vector values
            limit (int): All ids of the document to be queried
            offset (int): Page offset, used to control the starting position of the results
            filter (Union[Filter, str]): Filter condition of the scalar index field
            output_fields (List[str]): document's fields to return
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.
            sort: (dict): Set order by, like {'fieldName': 'age', 'direction': 'desc'}, default asc

        Returns:
            List[Dict]: all matched documents
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.query(
            document_ids=document_ids,
            retrieve_vector=retrieve_vector,
            limit=limit,
            offset=offset,
            filter=filter,
            output_fields=output_fields,
            timeout=timeout,
            sort=sort,
        )

    async def count(
        self,
        database_name: str,
        collection_name: str,
        filter: Union[Filter, str] = None,
        timeout: float = None,
    ) -> int:
        """Calculate the number of documents based on the query conditions.

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection.
            filter (Union[Filter, str]): The optional filter condition of the scalar index field.
            timeout (float): An optional duration of time in seconds to allow for the request.
                    When timeout is set to None, will use the connect timeout.

        Returns:
            int: The number of documents based on the query conditions
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.count(filter=filter, timeout=timeout)

    async def search(
        self,
        database_name: str,
        collection_name: str,
        vectors: Union[List[List[float]], ndarray],
        filter: Union[Filter, str] = None,
        params=None,
        retrieve_vector: bool = False,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Search the most similar vector by the given vectors. Batch API

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            vectors (Union[List[List[float]], ndarray]): The list of vectors
            filter (Union[Filter, str]): Filter condition of the scalar index field
            params (SearchParams): query parameters
                FLAT: No parameters need to be specified.
                HNSW: ef, specifying the number of vectors to be accessed. Value range [1,32768], default is 10.
                IVF series: nprobe, specifying the number of units to be queried. Value range [1,nlist].
            retrieve_vector (bool): Whether to return vector values
            limit (int): All ids of the document to be queried
            output_fields (List[str]): document's fields to return
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.
            radius (float): Based on the score threshold for similarity retrieval.
                            IP: return when score >= radius, value range (-∞, +∞).
                            COSINE: return when score >= radius, value range [-1, 1].
                            L2: return when score <= radius, value range [0, +∞).

        Returns:
            List[List[Dict]]: Return the most similar document for each vector.
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.search(
            vectors=vectors,
            filter=filter,
            params=params,
            retrieve_vector=retrieve_vector,
            limit=limit,
            output_fields=output_fields,
            timeout=timeout,
            radius=radius,
        )

    async def search_by_id(
        self,
        database_name: str,
        collection_name: str,
        document_ids: List[str],
        filter: Union[Filter, str] = None,
        params=None,
        retrieve_vector: bool = False,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Search the most similar vector by id. Batch API

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            document_ids (List[str]): The list of the document id
            filter (Union[Filter, str]): Filter condition of the scalar index field
            params (SearchParams): query parameters
                FLAT: No parameters need to be specified.
                HNSW: ef, specifying the number of vectors to be accessed. Value range [1,32768], default is 10.
                IVF series: nprobe, specifying the number of units to be queried. Value range [1,nlist].
            retrieve_vector (bool): Whether to return vector values
            limit (int): All ids of the document to be queried
            output_fields (List[str]): document's fields to return
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.
            radius (float): Based on the score threshold for similarity retrieval.
                            IP: return when score >= radius, value range (-∞, +∞).
                            COSINE: return when score >= radius, value range [-1, 1].
                            L2: return when score <= radius, value range [0, +∞).

        Returns:
            List[List[Dict]]: Return the most similar document for each id.
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.searchById(
            document_ids=document_ids,
            filter=filter,
            params=params,
            retrieve_vector=retrieve_vector,
            limit=limit,
            output_fields=output_fields,
            timeout=timeout,
            radius=radius,
        )

    async def search_by_text(
        self,
        database_name: str,
        collection_name: str,
        embedding_items: List[str],
        filter: Union[Filter, str] = None,
        params=None,
        retrieve_vector: bool = False,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search the most similar vector by the embeddingItem. Batch API
        The embedding_items will first be embedded into a vector by the model set by the collection on the server side.

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            embedding_items (Union[List[List[float]], ndarray]): The list of vectors
            filter (Union[Filter, str]): Filter condition of the scalar index field
            params (SearchParams): query parameters
                FLAT: No parameters need to be specified.
                HNSW: ef, specifying the number of vectors to be accessed. Value range [1,32768], default is 10.
                IVF series: nprobe, specifying the number of units to be queried. Value range [1,nlist].
            retrieve_vector (bool): Whether to return vector values
            limit (int): All ids of the document to be queried
            output_fields (List[str]): document's fields to return
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.
            radius (float): Based on the score threshold for similarity retrieval.
                            IP: return when score >= radius, value range (-∞, +∞).
                            COSINE: return when score >= radius, value range [-1, 1].
                            L2: return when score <= radius, value range [0, +∞).

        Returns:
            List[List[Dict]]: Return the most similar document for each embedding_item.
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.searchByText(
            embeddingItems=embedding_items,
            filter=filter,
            params=params,
            retrieve_vector=retrieve_vector,
            limit=limit,
            output_fields=output_fields,
            timeout=timeout,
            radius=radius,
        )

    async def hybrid_search(
        self,
        database_name: str,
        collection_name: str,
        ann: Optional[Union[List[AnnSearch], AnnSearch]] = None,
        match: Optional[Union[List[KeywordSearch], KeywordSearch]] = None,
        filter: Union[Filter, str] = None,
        rerank: Optional[Rerank] = None,
        retrieve_vector: Optional[bool] = None,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[List[Dict]]:
        """Dense Vector and Sparse Vector Hybrid Retrieval

        Args:
            database_name (str): The name of the database.
            collection_name (str): The name of the collection.
            ann (Union[List[AnnSearch], AnnSearch]): Sparse vector search params
            match (Union[List[KeywordSearch], KeywordSearch): Ann params for search
            filter (Union[Filter, str]): Filter condition of the scalar index field
            rerank (Rerank): rerank params, RRFRerank, WeightedRerank
            retrieve_vector (bool): Whether to return vector values
            limit (int): All ids of the document to be queried
            output_fields (List[str]): document's fields to return
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.

        Returns:
            Union[List[List[Dict], [List[Dict]]: Return the most similar document for each condition.
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.hybrid_search(
            ann=ann,
            match=match,
            filter=filter,
            rerank=rerank,
            retrieve_vector=retrieve_vector,
            output_fields=output_fields,
            limit=limit,
            timeout=timeout,
            **kwargs,
        )

    async def fulltext_search(
        self,
        database_name: str,
        collection_name: str,
        data: SparseVector,
        field_name: str = "sparse_vector",
        filter: Union[Filter, str] = None,
        retrieve_vector: Optional[bool] = None,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        terminate_after: Optional[int] = None,
        cutoff_frequency: Optional[float] = None,
        **kwargs,
    ) -> List[Dict]:
        """Sparse Vector retrieval

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection
            data (List[List[Union[int, float]]]): sparse vector to search.
            field_name (str): Sparse Vector field name, default: sparse_vector
            filter (Union[Filter, str]): The optional filter condition of the scalar index field.
            retrieve_vector (bool):  Whether to return vector values.
            output_fields (List[str]): document's fields to return.
            limit (int): return TopK=limit document.
            terminate_after(int): Set the upper limit for the number of retrievals.
                    This can effectively control the rate. For large datasets, the recommended empirical value is 4000.
            cutoff_frequency(float): Sets the upper limit for the frequency or occurrence count of high-frequency terms.
                    If the term frequency exceeds the value of cutoffFrequency, the keyword is ignored.

        Returns:
            [List[Dict]: the list of the matched document
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.fulltext_search(
            data=data,
            field_name=field_name,
            filter=filter,
            retrieve_vector=retrieve_vector,
            output_fields=output_fields,
            limit=limit,
            terminate_after=terminate_after,
            cutoff_frequency=cutoff_frequency,
            **kwargs,
        )

    async def rebuild_index(
        self,
        database_name: str,
        collection_name: str,
        drop_before_rebuild: bool = False,
        throttle: Optional[int] = None,
        timeout: Optional[float] = None,
        field_name: Optional[str] = None,
    ):
        """Rebuild all indexes under the specified collection.

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection
            drop_before_rebuild (bool): Whether to delete the old index before rebuilding the new index. Default False.
                                        true: first delete the old index and then rebuild the index.
                                        false: after creating the new index, then delete the old index.
            throttle (int): Whether to limit the number of CPU cores for building an index on a single node.
                            0: no limit.
            timeout (float): An optional duration of time in seconds to allow for the request.
                    When timeout is set to None, will use the connect timeout.
            field_name (str): Specify the fields for the reconstructed index.
                              One of vector or sparse_vector. Default vector.
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.rebuild_index(
            drop_before_rebuild=drop_before_rebuild,
            throttle=throttle,
            timeout=timeout,
            field_name=field_name,
        )

    async def add_index(
        self,
        database_name: str,
        collection_name: str,
        indexes: List[FilterIndex],
        build_existed_data: bool = True,
        timeout: Optional[float] = None,
    ) -> dict:
        """Add scalar field index to existing collection.

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection
            indexes (List[FilterIndex]): The scalar fields to add
            build_existed_data (bool): Whether scan historical Data and build index. Default is True.
                    If all fields are newly added, no need to scan historical data; can be set to False.
            timeout (float): An optional duration of time in seconds to allow for the request.
                    When timeout is set to None, will use the connect timeout.

        Returns:
            dict: The API returns a code and msg. For example: {"code": 0,  "msg": "Operation success"}
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.add_index(
            indexes=indexes, build_existed_data=build_existed_data, timeout=timeout
        )

    async def modify_vector_index(
        self,
        database_name: str,
        collection_name: str,
        vector_indexes: List[VectorIndex],
        rebuild_rules: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Adjust vector index parameters.

        Args:
            database_name (str): The name of the database where the collection resides.
            collection_name (str): The name of the collection
            vector_indexes (List[FilterIndex]): The vector fields to adjust
            rebuild_rules (dict): Specified rebuild rules.
                    This interface will trigger a rebuild after adjusting the parameters.
                    For example: {"drop_before_rebuild": True , "throttle": 1}
                    drop_before_rebuild (bool): Whether to delete the old index before rebuilding the new index during
                              index reconstruction. True: Delete the old index before rebuilding the index.
                    throttle (int): Whether to limit the number of CPU cores for building the index on a single node.
                              0: No limit on CPU cores. 1: CPU core count is 1.
            timeout (float): An optional duration of time in seconds to allow for the request.
                    When timeout is set to None, will use the connect timeout.

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "Start rebuilding. You can use the '/collection/describe' API to follow the progress of rebuilding."
           }
        """
        coll = await self.collection(database_name, collection_name)
        return await coll.modify_vector_index(
            vector_indexes=vector_indexes, rebuild_rules=rebuild_rules, timeout=timeout
        )

    async def create_user(self, user: str, password: str) -> dict:
        """Create a user.

        Args:
            user (str): The username to create.
            password (str): The password of user.

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "operation success"
           }
        """
        res = await self._conn.post(
            "/user/create", {"user": user, "password": password}
        )
        return res.data()

    async def drop_user(self, user: str) -> dict:
        """Drop a user.

        Args:
            user (str): The username to create.

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "operation success"
           }
        """
        res = await self._conn.post("/user/drop", {"user": user})
        return res.data()

    async def describe_user(self, user: str) -> dict:
        """Get a user info.

        Args:
            user (str): Username to get.

        Returns:
            dict: User info contains privileges. For example:
           {
              "user": "test_user",
              "createTime": "2024-10-01 00:00:00",
              "privileges": [
                {
                  "resource": "db0.*",
                  "actions": ["read"]
                }
              ]
            }
        """
        res = await self._conn.post("/user/describe", {"user": user})
        return res.data()

    async def user_list(self) -> List[dict]:
        """Get all users under the instance.

        Returns:
            dict: User info list. For example:
            [
              {
                "user": "test_user",
                "createTime": "2024-10-01 00:00:00",
                "privileges": [
                  {
                    "resource": "db0.*",
                    "actions": ["read"]
                  }
                ]
              }
           ]
        """
        res = await self._conn.post("/user/list", {})
        return res.data().get("users", [])

    async def change_password(self, user: str, password: str) -> dict:
        """Change password for user.

        Args:
            user (str): The user to change password.
            password (str): New password of the user.

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "operation success"
           }
        """
        res = await self._conn.post(
            "/user/changePassword", {"user": user, "password": password}
        )
        return res.data()

    async def grant_to_user(
        self, user: str, privileges: Union[dict, List[dict]]
    ) -> dict:
        """Grant permission for user.

        Args:
            user (str): The user to grant permission.
            privileges (str): The privileges to grant. For example:
            {
              "resource": "db0.*",
              "actions": ["read"]
            }

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "operation success"
           }
        """
        payload = {
            "user": user,
            "privileges": privileges if isinstance(privileges, list) else [privileges],
        }
        res = await self._conn.post("/user/grant", payload)
        return res.data()

    async def revoke_from_user(
        self, user: str, privileges: Union[dict, List[dict]]
    ) -> dict:
        """Revoke permission for user.

        Args:
            user (str): The user to revoke permission.
            privileges (str): The privilege to revoke. For example:
            {
              "resource": "db0.*",
              "actions": ["read"]
            }

        Returns:
            dict: The API returns a code and msg. For example:
           {
             "code": 0,
             "msg": "operation success"
           }
        """
        payload = {
            "user": user,
            "privileges": privileges if isinstance(privileges, list) else [privileges],
        }
        res = await self._conn.post("/user/revoke", payload)
        return res.data()
