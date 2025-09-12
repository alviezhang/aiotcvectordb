from typing import Optional, List, Union, Dict

import os
import asyncio

from aiotcvectordb.model.document_set import AsyncDocumentSet
from tcvectordb.model.collection_view import (
    SplitterProcess,
    CollectionView,
    Embedding,
    ParsingProcess,
)
from tcvectordb.model.document import Filter, Document
from tcvectordb.model.document_set import (
    Rerank,
    SearchResult,
    Chunk,
    DocumentSet,
    SearchParam,
    QueryParam,
)
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import Index
from tcvectordb.debug import Debug, Warning
from aiotcvectordb import exceptions
from qcloud_cos import CosConfig, CosS3Client


class AsyncCollectionView(CollectionView):
    """Async wrap of CollectionView"""

    def __init__(
        self,
        db,
        name: str,
        description: str = "",
        embedding: Optional[Embedding] = None,
        splitter_process: Optional[SplitterProcess] = None,
        index: Optional[Index] = None,
        expected_file_num: Optional[int] = None,
        average_file_size: Optional[int] = None,
        shard: Optional[int] = None,
        replicas: Optional[int] = None,
        parsing_process: Optional[ParsingProcess] = None,
    ):
        super().__init__(
            db,
            name,
            description,
            embedding,
            splitter_process,
            index,
            expected_file_num=expected_file_num,
            average_file_size=average_file_size,
            shard=shard,
            replicas=replicas,
            parsing_process=parsing_process,
        )

    async def load_and_split_text(
        self,
        local_file_path: str,
        document_set_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        splitter_process: Optional[SplitterProcess] = None,
        timeout: Optional[float] = None,
        parsing_process: Optional[ParsingProcess] = None,
    ) -> AsyncDocumentSet:
        # 基于 vendor 逻辑改为异步 HTTP 调用
        if not os.path.exists(local_file_path):
            raise exceptions.ParamError(message=f"file not found: {local_file_path}")
        if not os.path.isfile(local_file_path):
            raise exceptions.ParamError(message=f"not a file: {local_file_path}")
        # 校验与元数据
        self._chunk_splitter_check(local_file_path, splitter_process, parsing_process)
        cos_metadata = self._get_cos_metadata(
            metadata, splitter_process, parsing_process
        )
        _, file_name = os.path.split(local_file_path)
        if not document_set_name:
            document_set_name = file_name
        # 请求上传授权
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
            "documentSetName": document_set_name,
            "byteLength": os.stat(local_file_path).st_size,
        }
        if parsing_process:
            body["parsingProcess"] = vars(parsing_process)
        res = await self.db.conn.post("/ai/documentSet/uploadUrl", body, timeout)
        upload_condition = res.body.get("uploadCondition")
        credentials = res.body.get("credentials")
        if not upload_condition or not credentials:
            raise exceptions.ParamError(message="get file upload url failed")
        # 校验文件大小
        file_stat = os.stat(local_file_path)
        if file_stat.st_size == 0:
            raise exceptions.ParamError(
                message=f"{local_file_path} 0 bytes file denied"
            )
        if upload_condition.get("maxSupportContentLength", 0) < file_stat.st_size:
            raise exceptions.ParamError(
                message=f"{local_file_path} fileSize is invalid, support max content length is {upload_condition.get('maxSupportContentLength', 0)} bytes"
            )
        # 上传到 COS
        upload_path = res.body.get("uploadPath")
        cos_endpoint = res.body.get("cosEndpoint")
        bucket = (
            cos_endpoint.split(".")[0].replace("https://", "").replace("http://", "")
        )
        endpoint = cos_endpoint.split(".", 1)[1]
        config = CosConfig(
            Endpoint=endpoint,
            SecretId=credentials.get("TmpSecretId"),
            SecretKey=credentials.get("TmpSecretKey"),
            Token=credentials.get("Token"),
        )
        client = CosS3Client(config)
        document_set_id = res.body.get("documentSetId")
        cos_metadata["x-cos-meta-id"] = document_set_id
        cos_metadata["x-cos-meta-source"] = "PythonSDK"
        with open(local_file_path, "rb") as fp:
            response = await asyncio.to_thread(
                client.put_object,
                Bucket=bucket,
                Key=upload_path,
                Body=fp,
                Metadata=cos_metadata,
            )
        Debug("Put object response:")
        Debug(response)
        ds = DocumentSet(
            self,
            id=document_set_id,
            name=document_set_name,
            indexed_progress=0,
            indexed_status="New",
            splitter_process=splitter_process,
            parsing_process=parsing_process,
        )
        return ds_convert(ds)

    async def search(
        self,
        content: str,
        document_set_name: Optional[List[str]] = None,
        expand_chunk: Optional[list] = None,
        rerank: Optional[Rerank] = None,
        filter: Union[Filter, str] = None,
        limit: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[SearchResult]:
        search_param = SearchParam(
            content=content,
            document_set_name=document_set_name,
            expand_chunk=expand_chunk,
            rerank=rerank,
            filter=filter,
            limit=limit,
        )
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
            "search": vars(search_param),
        }
        res = await self.db.conn.post("/ai/documentSet/search", body, timeout)
        documents = res.body.get("documents", [])
        if not documents:
            return []
        return [SearchResult.from_dict(self, doc) for doc in documents]

    async def query(
        self,
        document_set_id: Optional[List] = None,
        document_set_name: Optional[List[str]] = None,
        filter: Union[Filter, str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[AsyncDocumentSet]:
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
        }
        query: Dict = {}
        if document_set_id is not None:
            query["documentSetId"] = document_set_id
        if document_set_name is not None:
            query["documentSetName"] = document_set_name
        if limit is not None:
            query["limit"] = limit
        if offset is not None:
            query["offset"] = offset
        if filter is not None:
            query["filter"] = filter if isinstance(filter, str) else filter.cond
        if output_fields:
            query["outputFields"] = output_fields
        body["query"] = query
        res = await self.db.conn.post("/ai/documentSet/query", body, timeout)
        documents = res.body.get("documentSets", [])
        if not documents:
            return []
        out: List[AsyncDocumentSet] = []
        for doc in documents:
            ds = DocumentSet(self, id=doc["documentSetId"], name=doc["documentSetName"])
            ds.load_fields(
                doc,
                self._parse_splitter_preprocess(doc),
                self._parse_parsing_process(doc),
            )
            out.append(ds_convert(ds))
        return out

    async def get_document_set(
        self,
        document_set_id: Optional[str] = None,
        document_set_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Union[AsyncDocumentSet, None]:
        if document_set_id is None and document_set_name is None:
            raise exceptions.ParamError(
                message="please provide document_set_id or document_set_name"
            )
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
            "documentSetName": document_set_name,
            "documentSetId": document_set_id,
        }
        res = await self.db.conn.post("/ai/documentSet/get", body, timeout)
        data = res.body.get("documentSet")
        if not data:
            return None
        ds = DocumentSet(self, id=data["documentSetId"], name=data["documentSetName"])
        ds.load_fields(
            data,
            self._parse_splitter_preprocess(data),
            self._parse_parsing_process(data),
        )
        return ds_convert(ds)

    async def delete(
        self,
        document_set_id: Union[str, List[str]] = None,
        document_set_name: Union[str, List[str]] = None,
        filter: Union[Filter, str] = None,
        timeout: float = None,
    ):
        if (not document_set_id) and (not document_set_name) and filter is None:
            raise exceptions.ParamError(
                message="please provide document_set_id or document_set_name or filter"
            )
        if document_set_id is not None and isinstance(document_set_id, str):
            document_set_id = [document_set_id]
        if document_set_name is not None and isinstance(document_set_name, str):
            document_set_name = [document_set_name]
        query = QueryParam(
            document_set_id=document_set_id,
            document_set_name=document_set_name,
            filter=filter,
        )
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
            "query": vars(query),
        }
        res = await self.db.conn.post("/ai/documentSet/delete", body, timeout)
        return res.data()

    async def update(
        self,
        data: Document,
        document_set_id: Union[str, List[str]] = None,
        document_set_name: Union[str, List[str]] = None,
        filter: Union[Filter, str] = None,
        timeout: float = None,
    ):
        if data is None:
            raise exceptions.ParamError(message="please provide update data")
        if (not document_set_id) and (not document_set_name) and filter is None:
            raise exceptions.ParamError(
                message="please provide document_set_id or document_set_name or filter"
            )
        if document_set_id is not None and isinstance(document_set_id, str):
            document_set_id = [document_set_id]
        if document_set_name is not None and isinstance(document_set_name, str):
            document_set_name = [document_set_name]
        query = QueryParam(
            document_set_id=document_set_id,
            document_set_name=document_set_name,
            filter=filter,
        )
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
            "query": vars(query),
            "update": vars(data),
        }
        res = await self.db.conn.post("/ai/documentSet/update", body, timeout)
        return res.data()

    async def get_chunks(
        self,
        document_set_id: Optional[str] = None,
        document_set_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[Chunk]:
        if (not document_set_id) and (not document_set_name):
            raise exceptions.ParamError(
                message="please provide document_set_id or document_set_name"
            )
        body = {
            "database": self.db.database_name,
            "collectionView": self.conn_name,
        }
        if document_set_id is not None:
            body["documentSetId"] = document_set_id
        if document_set_name is not None:
            body["documentSetName"] = document_set_name
        if limit is not None:
            body["limit"] = limit
        if offset is not None:
            body["offset"] = offset
        res = await self.db.conn.post("/ai/documentSet/getChunks", body, timeout)
        chunks = res.body.get("chunks", [])
        if not chunks:
            return []
        out: List[Chunk] = []
        for ck in chunks:
            chunk = Chunk(
                start_pos=ck.get("startPos"),
                end_pos=ck.get("endPos"),
                text=ck.get("text"),
            )
            out.append(chunk)
        return out

    async def upload_file(
        self,
        local_file_path: str,
        file_name: Optional[str] = None,
        splitter_process: Optional[SplitterProcess] = None,
        parsing_process: Optional[ParsingProcess] = None,
        embedding_model: Optional[str] = None,
        field_mappings: Optional[Dict[str, str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        # 文件检查
        if not os.path.exists(local_file_path):
            raise exceptions.ParamError(message=f"file not found: {local_file_path}")
        if not os.path.isfile(local_file_path):
            raise exceptions.ParamError(message=f"not a file: {local_file_path}")
        # 拆分器检查
        self._chunk_splitter_check(local_file_path, splitter_process)
        # 元数据
        cos_metadata = self._get_cos_metadata(metadata=metadata)
        _, f_name = os.path.split(local_file_path)
        if not file_name:
            file_name = f_name
        # 请求上传授权
        body = {
            "database": self.db.database_name,
            "collection": self.conn_name,
            "fileName": file_name,
            "byteLength": os.stat(local_file_path).st_size,
        }
        if splitter_process:
            body["splitterPreprocess"] = vars(splitter_process)
        if parsing_process:
            body["parsingProcess"] = vars(parsing_process)
        if embedding_model:
            body["embeddingModel"] = embedding_model
        if field_mappings:
            body["fieldMappings"] = field_mappings
        res = await self.db.conn.post("/ai/document/uploadUrl", body)
        upload_condition = res.body.get("uploadCondition")
        credentials = res.body.get("credentials")
        if not upload_condition or not credentials:
            raise exceptions.ParamError(message="get file upload url failed")
        # 文件大小检查
        file_stat = os.stat(local_file_path)
        if file_stat.st_size == 0:
            raise exceptions.ParamError(
                message=f"{local_file_path} 0 bytes file denied"
            )
        if upload_condition.get("maxSupportContentLength", 0) < file_stat.st_size:
            raise exceptions.ParamError(
                message=f"{local_file_path} fileSize is invalid, support max content length is {upload_condition.get('maxSupportContentLength', 0)} bytes"
            )
        warning = res.body.get("warning")
        if warning:
            Warning(warning)
        # 上传 COS
        upload_path = res.body.get("uploadPath")
        cos_endpoint = res.body.get("cosEndpoint")
        bucket = (
            cos_endpoint.split(".")[0].replace("https://", "").replace("http://", "")
        )
        endpoint = cos_endpoint.split(".", 1)[1]
        config = CosConfig(
            Endpoint=endpoint,
            SecretId=credentials.get("TmpSecretId"),
            SecretKey=credentials.get("TmpSecretKey"),
            Token=credentials.get("Token"),
        )
        client = CosS3Client(config)
        cos_metadata["x-cos-meta-source"] = "PythonSDK"
        with open(local_file_path, "rb") as fp:
            response = await asyncio.to_thread(
                client.put_object,
                Bucket=bucket,
                Key=upload_path,
                Body=fp,
                Metadata=cos_metadata,
            )
        Debug("Put cos object response:")
        Debug(response)
        body["id"] = file_name
        return body

    async def get_image_url(
        self, document_ids: List[str], file_name: str
    ) -> List[List[dict]]:
        body = {
            "database": self.db.database_name,
            "collection": self.conn_name,
            "documentIds": document_ids,
            "fileName": file_name,
        }
        res = await self.db.conn.post("/ai/document/getImageUrl", body)
        return res.data().get("images", [])

    async def query_file_details(
        self,
        database_name: str,
        collection_name: str,
        file_names: List[str] = None,
        filter: Union[Filter, str] = None,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        read_consistency: ReadConsistency = ReadConsistency.EVENTUAL_CONSISTENCY,
    ) -> List[Dict]:
        query: Dict = {}
        if file_names is not None:
            query["fileNames"] = file_names
        if filter is not None:
            query["filter"] = filter if isinstance(filter, str) else filter.cond
        if limit is not None:
            query["limit"] = limit
        if offset is not None:
            query["offset"] = offset
        if output_fields:
            query["outputFields"] = output_fields
        body = {
            "database": database_name,
            "collection": collection_name,
            "query": query,
            "readConsistency": read_consistency.value,
        }
        res = await self.db.conn.post("/ai/document/queryFileDetails", body)
        documents = res.body.get("documents", None)
        if not documents:
            return []
        return [doc for doc in documents]


def ds_convert(ds: DocumentSet) -> AsyncDocumentSet:
    return AsyncDocumentSet(
        collection_view=ds.collection_view,
        id=ds.id,
        name=ds.name,
        text_prefix=ds.text_prefix,
        text=ds.text,
        text_length=ds.document_set_info.text_length,
        byte_length=ds.document_set_info.byte_length,
        indexed_progress=ds.document_set_info.indexed_progress,
        indexed_status=ds.document_set_info.indexed_status,
        create_time=ds.document_set_info.create_time,
        last_update_time=ds.document_set_info.last_update_time,
        keywords=ds.document_set_info.keywords,
        indexed_error_msg=ds.document_set_info.indexed_error_msg,
        splitter_process=ds.splitter_process,
        parsing_process=ds.parsing_process,
        **ds.__getattribute__("_scalar_fields"),
    )
