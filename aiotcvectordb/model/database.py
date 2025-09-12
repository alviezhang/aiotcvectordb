from typing import List, Optional, Dict, Any, Union

from aiotcvectordb.model.ai_database import AsyncAIDatabase
from aiotcvectordb.model.collection import AsyncCollection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiotcvectordb.client.httpclient import AsyncHTTPClient
from tcvectordb.model.collection import Embedding, Collection, FilterIndexConfig
from tcvectordb.model.database import Database
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import Index, IndexField
from aiotcvectordb import exceptions as aio_exceptions
import tcvectordb.exceptions as vendor_exceptions


class AsyncDatabase(Database):
    """AsyncDatabase, Contains Database property and collection async API."""

    def __init__(
        self,
        conn: Union["AsyncHTTPClient", None],
        name: str = "",
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        info: Optional[dict] = None,
    ) -> None:
        super().__init__(conn, name, read_consistency, info=info)

    def __repr__(self) -> str:
        return (
            f"AsyncDatabase(name='{self.database_name}', "
            f"type='{self.db_type}', collections={self.collection_count})"
        )

    async def create_database(self, database_name="", timeout: Optional[float] = None):
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
        if not self.conn:
            raise aio_exceptions.NoConnectError
        if database_name:
            self._dbname = database_name
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database name param not found")
        body = {"database": self.database_name}
        await self.conn.post("/database/create", body, timeout)
        return self

    async def drop_database(
        self, database_name="", timeout: Optional[float] = None
    ) -> Dict:
        """Delete a database.

        Args:
            database_name (str): The name of the database to delete.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            Dict: Contains code、msg、affectedCount
        """
        if not self.conn:
            raise aio_exceptions.NoConnectError
        if database_name:
            self._dbname = database_name
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database name param not found")
        body = {"database": self.database_name}
        try:
            res = await self.conn.post("/database/drop", body, timeout)
            return res.data()
        except vendor_exceptions.VectorDBException as e:
            if e.message.find("not exist") == -1:
                raise aio_exceptions.ServerInternalError(code=e.code, message=e.message)

    async def list_databases(
        self, timeout: Optional[float] = None
    ) -> List[Union["AsyncDatabase", AsyncAIDatabase]]:
        """List all databases.

        Args:
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            List: all AsyncDatabase and AsyncAIDatabase
        """
        res = await self.conn.get("/database/list", timeout=timeout)
        databases = res.body.get("databases", [])
        db_info = res.body.get("info", {})
        out: List[Union["AsyncDatabase", AsyncAIDatabase]] = []
        for db_name in databases:
            info = db_info.get(db_name, {})
            db_type = info.get("dbType", "BASE_DB")
            if db_type in ("AI_DOC", "AI_DB"):
                out.append(
                    AsyncAIDatabase(
                        conn=self.conn,
                        name=db_name,
                        read_consistency=self._read_consistency,
                        info=info,
                    )
                )
            else:
                out.append(
                    AsyncDatabase(
                        conn=self.conn,
                        name=db_name,
                        read_consistency=self._read_consistency,
                        info=info,
                    )
                )
        return out

    async def exists_collection(self, collection_name: str) -> bool:
        """Check if the collection exists (async).

        Returns True if found, False if code 15302, else re-raises.
        """
        try:
            await self.describe_collection(collection_name)
            return True
        except vendor_exceptions.ServerInternalError as e:
            if e.code == 15302:
                return False
            raise aio_exceptions.ServerInternalError(code=e.code, message=e.message)

    async def create_collection(
        self,
        name: str,
        shard: int,
        replicas: int,
        description: str = None,
        index: Index = None,
        embedding: Embedding = None,
        timeout: float = None,
        ttl_config: dict = None,
        filter_index_config: FilterIndexConfig = None,
        indexes: List[IndexField] = None,
    ) -> AsyncCollection:
        """Create a collection.

        Args:
            name (str): The name of the collection. A collection name can only include
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

        Returns:
            A AsyncCollection object.
        """
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database not found")
        if not name:
            raise aio_exceptions.ParamError(message="collection name param not found")
        body = {
            "database": self.database_name,
            "collection": name,
            "shardNum": shard,
            "replicaNum": replicas,
        }
        # 与官方同步版一致，传递 embedding 字段
        body["embedding"] = vars(embedding) if embedding else {}
        if description is not None:
            body["description"] = description
        # 支持 index 或 indexes 两种传参方式
        if index is None and indexes:
            index = Index()
            for idx in indexes:
                index.add(idx)
        if index is not None:
            body["indexes"] = index.list()
        if ttl_config is not None:
            body["ttlConfig"] = ttl_config
        if filter_index_config is not None:
            body["filterIndexConfig"] = vars(filter_index_config)
        await self.conn.post("/collection/create", body, timeout)
        coll = Collection(
            self,
            name,
            shard,
            replicas,
            description,
            index,
            embedding=embedding,
            ttl_config=ttl_config,
            filter_index_config=filter_index_config,
            read_consistency=self._read_consistency,
        )
        return coll_convert(coll)

    async def create_collection_if_not_exists(
        self,
        name: str,
        shard: int,
        replicas: int,
        description: str = None,
        index: Index = None,
        embedding: Embedding = None,
        timeout: float = None,
        ttl_config: dict = None,
        filter_index_config: FilterIndexConfig = None,
        indexes: List[IndexField] = None,
    ) -> AsyncCollection:
        """Create the collection if it doesn't exist.

        Args:
            name (str): The name of the collection. A collection name can only include
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

        Returns:
            AsyncCollection: A collection object.
        """
        try:
            return await self.describe_collection(name, timeout)
        except vendor_exceptions.ServerInternalError as e:
            if e.code != 15302:
                raise aio_exceptions.ServerInternalError(code=e.code, message=e.message)
        await self.create_collection(
            name=name,
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
        return await self.describe_collection(name, timeout)

    async def list_collections(
        self, timeout: Optional[float] = None
    ) -> List[AsyncCollection]:
        """List all collections in the database.

        Args:
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            List: all AsyncCollection
        """
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database not found")
        body = {"database": self.database_name}
        res = await self.conn.post("/collection/list", body, timeout)
        collections: List[AsyncCollection] = []
        for col in res.body["collections"]:
            coll = _gen_collection(self, col, self._read_consistency)
            acoll = coll_convert(coll)
            collections.append(acoll)
        return collections

    async def collection(self, name: str) -> AsyncCollection:
        """Get a Collection by name (async)."""
        return await self.describe_collection(name)

    async def describe_collection(
        self, name: str, timeout: Optional[float] = None
    ) -> AsyncCollection:
        """Get a Collection by name.

        Args:
            name (str): The name of the collection.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            A AsyncCollection object.
        """
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database not found")
        if not name:
            raise aio_exceptions.ParamError(message="collection name param not found")
        body = {"database": self.database_name, "collection": name}
        res = await self.conn.post("/collection/describe", body, timeout)
        if not res.body["collection"]:
            raise aio_exceptions.DescribeCollectionException(
                code=-1, message=str(res.body)
            )
        col = res.body["collection"]
        coll = _gen_collection(self, col, self._read_consistency)
        acoll = coll_convert(coll)
        acoll.conn_name = name
        return acoll

    async def drop_collection(self, name: str, timeout: Optional[float] = None) -> Dict:
        """Delete a collection by name.

        Args:
            name (str): The name of the collection.
            timeout (float): An optional duration of time in seconds to allow for the request. When timeout
                is set to None, will use the connect timeout.

        Returns:
            Dict: Contains code、msg、affectedCount
        """
        # use module-level aliases
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database not found")
        if not name:
            raise aio_exceptions.ParamError(message="collection name param not found")
        body = {"database": self.database_name, "collection": name}
        try:
            res = await self.conn.post("/collection/drop", body, timeout)
            return res.data()
        except vendor_exceptions.VectorDBException as e:
            if e.message.find("not exist") == -1:
                raise aio_exceptions.ServerInternalError(code=e.code, message=e.message)

    async def truncate_collection(self, collection_name: str) -> Dict:
        """Clear all the data and indexes in the Collection.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Dict: Contains affectedCount
        """
        if not self.database_name:
            raise aio_exceptions.ParamError(message="param database is blank")
        if not collection_name:
            raise aio_exceptions.ParamError(message="collection name param not found")
        body = {"database": self.database_name, "collection": collection_name}
        res = await self.conn.post("/collection/truncate", body)
        return res.data()

    async def set_alias(self, collection_name: str, collection_alias: str) -> Dict:
        """Set alias for collection.

        Args:
            collection_name  : The name of the collection.
            collection_alias : alias name to set.
        Returns:
            Dict: Contains affectedCount
        """
        if not self.database_name:
            raise aio_exceptions.ParamError(message="database not found")
        if not collection_name:
            raise aio_exceptions.ParamError(message="collection name param not found")
        if not collection_alias:
            raise aio_exceptions.ParamError(message="collection_alias is blank")
        body = {
            "database": self.database_name,
            "collection": collection_name,
            "alias": collection_alias,
        }
        postRes = await self.conn.post("/alias/set", body)
        if "affectedCount" in postRes.body:
            return {"affectedCount": postRes.body.get("affectedCount")}
        raise aio_exceptions.ServerInternalError(
            message=f"response content is not as expected: {postRes.body}"
        )

    async def delete_alias(self, alias: str) -> Dict[str, Any]:
        """Delete alias by name.

        Args:
            alias  : alias name to delete.

        Returns:
            Dict: Contains affectedCount
        """
        if not self.database_name or not alias:
            raise aio_exceptions.ParamError(message="database and alias required")
        body = {"database": self.database_name, "alias": alias}
        postRes = await self.conn.post("/alias/delete", body)
        if "affectedCount" in postRes.body:
            return {"affectedCount": postRes.body.get("affectedCount")}
        raise aio_exceptions.ServerInternalError(
            message=f"response content is not as expected: {postRes.body}"
        )


def _gen_collection(
    db: Database, col: Dict[str, Any], read_consistency: ReadConsistency
) -> Collection:
    index = Index()
    for elem in col.pop("indexes", []):
        index.add(**elem)
    ebd = None
    if "embedding" in col:
        ebd = Embedding()
        ebd.set_fields(**col.pop("embedding", {}))
    filter_index_config = None
    if "filterIndexConfig" in col:
        filter_index_config = FilterIndexConfig(**col.pop("filterIndexConfig", {}))
    collection = Collection(
        db,
        name=col.pop("collection", None),
        shard=col.pop("shardNum", None),
        replicas=col.pop("replicaNum", None),
        description=col.pop("description", None),
        index=index,
        embedding=ebd,
        ttl_config=col.pop("ttlConfig", None),
        filter_index_config=filter_index_config,
        read_consistency=read_consistency,
        **col,
    )
    return collection


def db_convert(db) -> Union[AsyncDatabase, AsyncAIDatabase]:
    read_consistency = db.__getattribute__("_read_consistency")
    if isinstance(db, Database):
        return AsyncDatabase(
            conn=db.conn,
            name=db.database_name,
            read_consistency=read_consistency,
            info=db.info,
        )
    else:
        return AsyncAIDatabase(
            conn=db.conn,
            name=db.database_name,
            read_consistency=read_consistency,
            info=db.info,
        )


def coll_convert(coll: Collection) -> AsyncCollection:
    read_consistency = coll.__getattribute__("_read_consistency")
    a_coll = AsyncCollection(
        db=AsyncDatabase(
            conn=coll.__getattribute__("_conn"),
            name=coll.database_name,
            read_consistency=read_consistency,
        ),
        name=coll.collection_name,
        shard=coll.shard,
        replicas=coll.replicas,
        description=coll.description,
        index=coll.index,
        embedding=coll.embedding,
        read_consistency=read_consistency,
        ttl_config=coll.ttl_config,
        filter_index_config=coll.filter_index_config,
        createTime=coll.create_time,
        documentCount=coll.document_count,
        alias=coll.alias,
        indexStatus=coll.index_status,
        **coll.kwargs,
    )
    return a_coll
