from __future__ import annotations
from typing import Dict, List, Optional, Any, Union

from numpy import ndarray

from tcvectordb.model.collection import (
    Collection,
    FilterIndexConfig,
    Query,
    Search,
    DeleteQuery,
    UpdateQuery,
)
from tcvectordb.model.collection_view import Embedding
from tcvectordb.model.document import Document, Filter, AnnSearch, KeywordSearch, Rerank
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import Index, SparseVector, FilterIndex, VectorIndex
from tcvectordb.debug import Warning
from aiotcvectordb import exceptions as aio_exceptions


class AsyncCollection(Collection):
    """AsyncCollection

    Contains Collection property and document API..

    Args:
        db (Database): Database object.
        name (str): collection name.
        shard (int): The shard number of the collection.
        replicas (int): The replicas number of the collection.
        description (str): An optional description of the collection.
        index (Index): A list of the index properties for the documents in a collection.
        read_consistency (ReadConsistency): STRONG_CONSISTENCY or EVENTUAL_CONSISTENCY for query
        embedding (Embedding): An optional embedding for embedding text when upsert documents.
        ttl_config (dict): TTL configuration, when set {'enable': True, 'timeField': 'expire_at'} means
            that ttl is enabled and automatically removed when the time set in the expire_at field expires
        filter_index_config (FilterIndexConfig): Enabling full indexing mode.
            Where all scalar fields are indexed by default.
        kwargs:
            create_time(str): collection create time
    """

    def __init__(
        self,
        db,
        name: str = "",
        shard=0,
        replicas=0,
        description="",
        index: Index = None,
        embedding: Embedding = None,
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        ttl_config: dict = None,
        filter_index_config: FilterIndexConfig = None,
        **kwargs,
    ):
        super().__init__(
            db,
            name,
            shard,
            replicas,
            description,
            index,
            embedding,
            read_consistency,
            ttl_config=ttl_config,
            filter_index_config=filter_index_config,
            **kwargs,
        )

    def __repr__(self) -> str:
        # 下面这些字段在父类 Collection.__init__ 中都会初始化
        return (
            f"AsyncCollection(db='{self.database_name}', "
            f"name='{self.collection_name}', shards={self.shard}, replicas={self.replicas})"
        )

    async def upsert(
        self,
        documents: List[Union[Document, Dict]],
        timeout: Optional[float] = None,
        build_index: bool = True,
        **kwargs,
    ):
        """Upsert documents into a collection.

        Args:
            documents (List[Union[Document, Dict]]) : The list of the document object or dict to upsert. Maximum 1000.
            timeout (float) : An optional duration of time in seconds to allow for the request.
                              When timeout is set to None, will use the connect timeout.
            build_index (bool) : An option for build index time when upsert, if build_index is true, will build index
                                 immediately, it will affect performance of upsert. And param buildIndex has same
                                 semantics with build_index, any of them false will be false

        Returns:
            Dict: Contains affectedCount
        """
        buildIndex = bool(kwargs.get("buildIndex", True))
        res_build_index = buildIndex and build_index
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "buildIndex": res_build_index,
            "documents": [],
        }
        ai = False
        if len(documents) > 0:
            if isinstance(documents[0], dict):
                ai = isinstance(documents[0].get("vector"), str)
            else:
                ai = isinstance(vars(documents[0]).get("vector"), str)
        for doc in documents:
            if isinstance(doc, dict):
                body["documents"].append(doc)
            else:
                body["documents"].append(vars(doc))
        res = await self._conn.post("/document/upsert", body, timeout, ai=ai)
        return res.data()

    async def query(
        self,
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
        query_param = Query(
            limit=limit,
            offset=offset,
            retrieve_vector=retrieve_vector,
            filter=filter,
            document_ids=document_ids,
            output_fields=output_fields,
            sort=sort,
        )
        return await self.__base_query_async(
            query=query_param, read_consistency=self._read_consistency, timeout=timeout
        )

    async def search(
        self,
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
        search_param = Search(
            retrieve_vector=retrieve_vector,
            limit=limit,
            vectors=vectors,
            filter=filter,
            params=params,
            output_fields=output_fields,
            radius=radius,
        )
        res = await self.__base_search_async(
            search=search_param,
            read_consistency=self._read_consistency,
            timeout=timeout,
        )
        return res.get("documents")

    async def searchById(
        self,
        document_ids: List,
        filter: Union[Filter, str] = None,
        params=None,
        retrieve_vector: bool = False,
        limit: int = 10,
        timeout: Optional[float] = None,
        output_fields: Optional[List[str]] = None,
        radius: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Search the most similar vector by id. Batch API

        Args:
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
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        search_param = Search(
            retrieve_vector=retrieve_vector,
            limit=limit,
            document_ids=document_ids,
            filter=filter,
            params=params,
            output_fields=output_fields,
            radius=radius,
        )
        res = await self.__base_search_async(
            search=search_param,
            read_consistency=self._read_consistency,
            timeout=timeout,
        )
        return res.get("documents")

    async def searchByText(
        self,
        embeddingItems: List[str],
        filter: Union[Filter, str] = None,
        params=None,
        retrieve_vector: bool = False,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search the most similar vector by the embeddingItem. Batch API
        The embeddingItem will first be embedded into a vector by the model set by the collection on the server side.

        Args:
            embeddingItems (Union[List[List[float]], ndarray]): The list of vectors
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
            List[List[Dict]]: Return the most similar document for each embeddingItem.
        """
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        search_param = Search(
            retrieve_vector=retrieve_vector,
            limit=limit,
            embedding_items=embeddingItems,
            filter=filter,
            params=params,
            output_fields=output_fields,
            radius=radius,
        )
        return await self.__base_search_async(
            search=search_param,
            read_consistency=self._read_consistency,
            timeout=timeout,
        )

    async def hybrid_search(
        self,
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
        single = True
        if ann:
            if isinstance(ann, List):
                single = False
            else:
                ann = [ann]
        if match:
            if isinstance(match, List):
                single = False
            else:
                match = [match]
        search: Dict[str, Any] = {}
        ai = False
        if ann:
            search["ann"] = []
            for a in ann:
                search["ann"].append(vars(a))
            if len(ann) > 0 and ann[0].data is not None:
                if isinstance(ann[0].data, str):
                    ai = True
                elif len(ann[0].data) > 0 and isinstance(ann[0].data[0], str):
                    ai = True
        if match:
            search["match"] = []
            for m in match:
                search["match"].append(vars(m))
        if filter:
            search["filter"] = filter if isinstance(filter, str) else filter.cond
        if rerank:
            search["rerank"] = vars(rerank)
        if retrieve_vector is not None:
            search["retrieveVector"] = retrieve_vector
        if output_fields:
            search["outputFields"] = output_fields
        if limit is not None:
            search["limit"] = limit
        search.update(kwargs)
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "readConsistency": self._read_consistency.value,
            "search": search,
        }
        res = await self._conn.post("/document/hybridSearch", body, timeout, ai=ai)
        if "warning" in res.body:
            Warning(res.body.get("warning"))
        documents = res.body.get("documents", None)
        if not documents:
            return []
        documents_res: List[List[Dict]] = []
        for arr in documents:
            tmp: List[Dict] = []
            for elem in arr:
                tmp.append(elem)
            documents_res.append(tmp)
        if single:
            documents_res = documents_res[0]
        return documents_res

    async def fulltext_search(
        self,
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
        match = {"fieldName": field_name, "data": [data]}
        if terminate_after is not None:
            match["terminateAfter"] = terminate_after
        if cutoff_frequency is not None:
            match["cutoffFrequency"] = cutoff_frequency
        search = {"match": match}
        if filter:
            search["filter"] = filter if isinstance(filter, str) else filter.cond
        if retrieve_vector is not None:
            search["retrieveVector"] = retrieve_vector
        if output_fields:
            search["outputFields"] = output_fields
        if limit is not None:
            search["limit"] = limit
        search.update(kwargs)
        body = {
            "database": self.database_name,
            "collection": self.collection_name,
            "readConsistency": self._read_consistency.value,
            "search": search,
        }
        res = await self._conn.post("/document/fullTextSearch", body)
        if "warning" in res.body:
            Warning(res.body.get("warning"))
        documents = res.body.get("documents", None)
        if not documents:
            return []
        documents_res: List[List[Dict]] = []
        for arr in documents:
            tmp: List[Dict] = []
            for elem in arr:
                tmp.append(elem)
            documents_res.append(tmp)
        return documents_res[0]

    async def delete(
        self,
        document_ids: List[str] = None,
        filter: Union[Filter, str] = None,
        timeout: float = None,
        limit: Optional[int] = None,
    ) -> Dict:
        """Delete document by conditions.

        Args:
            document_ids (List[str]): The list of the document id
            filter (Union[Filter, str]): Filter condition of the scalar index field
            limit (int): The amount of document deleted, with a range of [1, 16384].
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.

        Returns:
            Dict: Contains affectedCount
        """
        delete_query_param = DeleteQuery(
            filter=filter, document_ids=document_ids, limit=limit
        )
        return await self.__base_delete_async(
            delete_query=delete_query_param, timeout=timeout
        )

    async def update(
        self,
        data: Union[Document, Dict],
        filter: Union[Filter, str] = None,
        document_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict:
        """Update document by conditions.

        Args:
            data (Union[Document, Dict]): Set the fields to be updated.
            document_ids (List[str]): The list of the document id
            filter (Union[Filter, str]): Filter condition of the scalar index field
            timeout (float): An optional duration of time in seconds to allow for the request.
                             When timeout is set to None, will use the connect timeout.

        Returns:
            Dict: Contains affectedCount
        """

        if data is None:
            raise aio_exceptions.ParamError(code=-1, message="data is None")
        update_query = UpdateQuery(document_ids=document_ids, filter=filter)
        return await self.__base_update_async(
            update_query=update_query, document=data, timeout=timeout
        )

    async def rebuild_index(
        self,
        drop_before_rebuild: bool = False,
        throttle: Optional[int] = None,
        timeout: Optional[float] = None,
        field_name: Optional[str] = None,
    ):
        """Rebuild all indexes under the specified collection.

        Args:
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

        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        body = {
            "database": self.database_name,
            "collection": self.collection_name,
            "dropBeforeRebuild": drop_before_rebuild,
        }
        if throttle is not None:
            body["throttle"] = throttle
        if field_name is not None:
            body["fieldName"] = field_name
        await self._conn.post("/index/rebuild", body, timeout)

    async def count(
        self, filter: Union[Filter, str] = None, timeout: float = None
    ) -> int:
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
        }
        if self._read_consistency is not None:
            body["readConsistency"] = self._read_consistency.value
        query = {}
        if filter is not None:
            query["filter"] = filter if isinstance(filter, str) else filter.cond
        body["query"] = query
        res = await self._conn.post("/document/count", body, timeout)
        return res.data().get("count")

    async def add_index(
        self,
        indexes: List[FilterIndex],
        build_existed_data: bool = True,
        timeout: Optional[float] = None,
    ) -> dict:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        indexes_payload = [vars(item) for item in indexes]
        body = {
            "database": self.database_name,
            "collection": self.collection_name,
            "indexes": indexes_payload,
        }
        if build_existed_data is not None:
            body["buildExistedData"] = build_existed_data
        res = await self._conn.post("/index/add", body, timeout)
        return res.data()

    async def drop_index(
        self, field_names: List[str], timeout: Optional[float] = None
    ) -> dict:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        if not isinstance(field_names, list):
            raise aio_exceptions.ParamError(
                message="Invalid value for List[str] field: field_names"
            )
        body = {
            "database": self.database_name,
            "collection": self.collection_name,
            "fieldNames": field_names,
        }
        res = await self._conn.post("/index/drop", body, timeout)
        return res.data()

    async def modify_vector_index(
        self,
        vector_indexes: List[VectorIndex],
        rebuild_rules: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        indexes = []
        for item in vector_indexes:
            index = vars(item)
            if hasattr(item, "field_type_none") and item.field_type_none:
                del index["fieldType"]
            indexes.append(index)
        body = {
            "database": self.database_name,
            "collection": self.collection_name,
            "vectorIndexes": indexes,
        }
        if rebuild_rules is not None:
            if "drop_before_rebuild" in rebuild_rules:
                rebuild_rules["dropBeforeRebuild"] = rebuild_rules.pop(
                    "drop_before_rebuild"
                )
            body["rebuildRules"] = rebuild_rules
        res = await self._conn.post("/index/modifyVectorIndex", body, timeout)
        return res.data()

    async def __base_query_async(
        self,
        query: Query,
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        timeout: Optional[float] = None,
    ) -> List[Dict]:
        if query is None:
            raise aio_exceptions.ParamError(
                code=-1, message="query is a required parameter"
            )
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "query": vars(query),
            "readConsistency": read_consistency.value,
        }
        res = await self._conn.post("/document/query", body, timeout)
        documents = res.body.get("documents", None)
        if not documents:
            return []
        return [doc for doc in documents]

    async def __base_search_async(
        self,
        search: Search,
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        if search is None:
            raise aio_exceptions.ParamError(message="search is None")
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "readConsistency": read_consistency.value,
            "search": vars(search),
        }
        ai = False
        if (
            isinstance(search.vectors, list)
            and len(search.vectors) > 0
            and isinstance(search.vectors[0], str)
        ):
            ai = True
        res = await self._conn.post("/document/search", body, timeout, ai=ai)
        warn_msg = ""
        if (
            res.body.get("warning", None) is not None
            and len(res.body.get("warning", None)) > 0
        ):
            warn_msg = res.body.get("warning")
        documents = res.body.get("documents", None)
        if not documents:
            return {"warning": warn_msg, "documents": []}
        documents_res: List[List[Dict]] = []
        for arr in documents:
            tmp: List[Dict] = []
            for elem in arr:
                tmp.append(elem)
            documents_res.append(tmp)
        return {"warning": warn_msg, "documents": documents_res}

    async def __base_delete_async(
        self, delete_query: DeleteQuery, timeout: Optional[float] = None
    ) -> Dict:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "query": vars(delete_query),
        }
        res = await self._conn.post("/document/delete", body, timeout)
        return res.data()

    async def __base_update_async(
        self,
        update_query: UpdateQuery,
        document: Union[Document, Dict],
        timeout: Optional[float] = None,
    ) -> Dict:
        if not self.database_name or not self.collection_name:
            raise aio_exceptions.ParamError(
                message="database_name or collection_name is blank"
            )
        if update_query is None or not update_query.valid():
            raise aio_exceptions.ParamError(
                code=-1, message="query both field document_ids and filter are None"
            )
        if document is None:
            raise aio_exceptions.ParamError(code=-1, message="document is None")
        body = {
            "database": self.database_name,
            "collection": self.conn_name,
            "query": vars(update_query),
        }
        ai = False
        if isinstance(document, dict):
            ai = isinstance(document.get("vector"), str)
        else:
            ai = isinstance(vars(document).get("vector"), str)
        body["update"] = document if isinstance(document, dict) else vars(document)
        postRes = await self._conn.post("/document/update", body, timeout, ai=ai)
        resBody = postRes.body
        res: Dict[str, Any] = {}
        if "warning" in resBody:
            res["warning"] = resBody.get("warning")
        if "affectedCount" in resBody:
            res["affectedCount"] = resBody.get("affectedCount")
        return res
