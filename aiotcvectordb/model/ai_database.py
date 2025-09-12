from typing import Optional, List, Dict, Any

from aiotcvectordb.model.collection_view import AsyncCollectionView
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiotcvectordb.client.httpclient import AsyncHTTPClient
from tcvectordb.model.ai_database import AIDatabase
from tcvectordb.model.collection_view import (
    SplitterProcess,
    Embedding,
    CollectionView,
    ParsingProcess,
)
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import Index
from aiotcvectordb import exceptions as aio_exceptions


class AsyncAIDatabase(AIDatabase):
    """Async wrap of AIDatabase"""

    def __init__(
        self,
        conn: "AsyncHTTPClient",
        name: str,
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
        info: Optional[dict] = None,
    ):
        super().__init__(conn, name, read_consistency, info=info)

    def __repr__(self) -> str:
        return (
            f"AsyncAIDatabase(name='{self.database_name}', "
            f"type='{self.db_type}', collections={self.collection_count})"
        )

    async def create_database(self, database_name="", timeout: Optional[float] = None):
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
        if database_name:
            self.database_name = database_name
        body = {"database": self.database_name}
        await self.conn.post("/ai/database/create", body, timeout)
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
        if database_name:
            self.database_name = database_name
        body = {"database": self.database_name}
        res = await self.conn.post("/ai/database/drop", body, timeout)
        return res.data()

    async def create_collection_view(
        self,
        name: str,
        description: str = "",
        embedding: Optional[Embedding] = None,
        splitter_process: Optional[SplitterProcess] = None,
        index: Optional[Index] = None,
        timeout: Optional[float] = None,
        expected_file_num: Optional[int] = None,
        average_file_size: Optional[int] = None,
        shard: Optional[int] = None,
        replicas: Optional[int] = None,
        parsing_process: Optional[ParsingProcess] = None,
    ) -> AsyncCollectionView:
        """Create a collection view.

        Args:
            name            : The name of the collection view.
            description     : An optional description of the collection view.
            embedding       : Args for embedding.
            splitter_process: Args for splitter process
            index           : A list of the index properties for the documents in a collection.
            timeout         : An optional duration of time in seconds to allow for the request.
                              When timeout is set to None, will use the connect timeout.
            expected_file_num: Expected total number of documents
            average_file_size: Estimate the average document size
            shard            : The shard number of the collection.
                               Shard will divide a large dataset into smaller subsets.
            replicas         : The replicas number of the collection.
                               Replicas refers to the number of identical copies of each primary shard,
                               used for disaster recovery and load balancing.
            parsing_process  : Document parsing parameters
        Returns:
            A AsyncCollectionView object
        """
        coll = CollectionView(
            db=self,
            name=name,
            description=description,
            embedding=embedding,
            splitter_process=splitter_process,
            index=index,
            expected_file_num=expected_file_num,
            average_file_size=average_file_size,
            shard=shard,
            replicas=replicas,
            parsing_process=parsing_process,
        )
        await self.conn.post("/ai/collectionView/create", vars(coll), timeout)
        return cv_convert(coll)

    async def describe_collection_view(
        self, collection_view_name: str, timeout: Optional[float] = None
    ) -> AsyncCollectionView:
        """Get a CollectionView by name.

        Args:
            collection_view_name: The name of the collection view
            timeout             : An optional duration of time in seconds to allow for the request.
                                  When timeout is set to None, will use the connect timeout.
        Returns:
            A AsyncCollectionView object
        """
        if not collection_view_name:
            raise aio_exceptions.ParamError(
                message="collection_view_name param not found"
            )
        body = {
            "database": self.database_name,
            "collectionView": collection_view_name,
        }
        res = await self.conn.post("/ai/collectionView/describe", body, timeout)
        if not res.body["collectionView"]:
            raise aio_exceptions.DescribeCollectionException(
                code=-1, message=str(res.body)
            )
        col = res.body["collectionView"]
        cv = CollectionView(self, col["collectionView"])
        cv.load_fields(col)
        acv = cv_convert(cv)
        acv.conn_name = collection_view_name
        return acv

    async def list_collection_view(
        self, timeout: Optional[float] = None
    ) -> List[AsyncCollectionView]:
        """Get collection view list.

        Args:
            timeout         : An optional duration of time in seconds to allow for the request.
                              When timeout is set to None, will use the connect timeout.
        Returns:
            List: all AsyncCollectionView objects
        """
        body = {"database": self.database_name}
        res = await self.conn.post("/ai/collectionView/list", body, timeout)
        collections: List[AsyncCollectionView] = []
        for col in res.body["collectionViews"]:
            cv = CollectionView(self, col["collectionView"])
            cv.load_fields(col)
            collections.append(cv_convert(cv))
        return collections

    async def collection_view(
        self, collection_view_name: str, timeout: Optional[float] = None
    ) -> AsyncCollectionView:
        """Get a CollectionView by name.

        Args:
            collection_view_name (str): The name of the CollectionView .
            timeout (float) : An optional duration of time in seconds to allow for the request.
                              When timeout is set to None, will use the connect timeout.

        Returns:
            A CollectionView object
        """
        return await self.describe_collection_view(collection_view_name, timeout)

    async def drop_collection_view(
        self, collection_view_name: str, timeout: Optional[float] = None
    ) -> Dict:
        """Delete a CollectionView by name.

        Args:
            collection_view_name: The name of the collection view
            timeout             : An optional duration of time in seconds to allow for the request.
                                  When timeout is set to None, will use the connect timeout.
        Returns:
            Dict: Contains code、msg、affectedCount
        """
        if not collection_view_name:
            raise aio_exceptions.ParamError(
                message="collection_view_name param not found"
            )
        body = {
            "database": self.database_name,
            "collectionView": collection_view_name,
        }
        res = await self.conn.post("/ai/collectionView/drop", body, timeout)
        return res.data()

    async def truncate_collection_view(
        self, collection_view_name: str, timeout: Optional[float] = None
    ) -> Dict:
        """Clear all data and indexes in the collection view.

        Args:
            collection_view_name: The name of the collection view
            timeout             : An optional duration of time in seconds to allow for the request.
                                  When timeout is set to None, will use the connect timeout.
        Returns:
            Dict: Contains affectedCount
        """
        if not collection_view_name:
            raise aio_exceptions.ParamError(
                message="collection_view_name param not found"
            )
        body = {
            "database": self.database_name,
            "collectionView": collection_view_name,
        }
        res = await self.conn.post("/ai/collectionView/truncate", body, timeout)
        return res.data()

    async def set_alias(
        self,
        collection_view_name: str,
        alias: str,
    ) -> Dict[str, Any]:
        """Set alias for collection view.

        Args:
            collection_view_name: The name of the collection_view.
            alias               : alias name to set.

        Returns:
            Dict: Contains affectedCount
        """
        if not collection_view_name:
            raise aio_exceptions.ParamError(
                message="collection_view_name param not found"
            )
        if not alias:
            raise aio_exceptions.ParamError(message="alias param not found")
        body = {
            "database": self.database_name,
            "collectionView": collection_view_name,
            "alias": alias,
        }
        res = await self.conn.post("/ai/alias/set", body)
        return res.data()

    async def delete_alias(self, alias: str) -> Dict[str, Any]:
        """Delete alias by name.

        Args:
            alias  : alias name to delete.

        Returns:
            Dict: Contains affectedCount
        """
        if not alias:
            raise aio_exceptions.ParamError(message="alias param not found")
        body = {"database": self.database_name, "alias": alias}
        res = await self.conn.post("/ai/alias/delete", body)
        return res.data()


def cv_convert(coll: CollectionView) -> AsyncCollectionView:
    return AsyncCollectionView(
        db=coll.db,
        name=coll.name,
        description=coll.description,
        embedding=coll.embedding,
        splitter_process=coll.splitter_process,
        index=coll.index,
        expected_file_num=coll.expected_file_num,
        average_file_size=coll.average_file_size,
        shard=coll.shard,
        replicas=coll.replicas,
        parsing_process=coll.parsing_process,
    )
