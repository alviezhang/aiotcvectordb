import os
import uuid
import pytest
import pytest_asyncio

from aiotcvectordb import AsyncVectorDBClient
from aiotcvectordb.model import (
    FieldType,
    IndexType,
    MetricType,
    ReadConsistency,
    Index,
    VectorIndex,
    FilterIndex,
    CollectionEmbedding as Embedding,
)


def _load_env():
    url = os.getenv("TCVECTORDB_URL")
    username = os.getenv("TCVECTORDB_USERNAME")
    key = os.getenv("TCVECTORDB_KEY")
    database = os.getenv("TCVECTORDB_DATABASE")
    missing = [
        var
        for var, val in (
            ("TCVECTORDB_URL", url),
            ("TCVECTORDB_USERNAME", username),
            ("TCVECTORDB_KEY", key),
            ("TCVECTORDB_DATABASE", database),
        )
        if not val
    ]
    if missing:
        pytest.fail(
            "Missing required environment variable(s): " + ", ".join(missing),
            pytrace=False,
        )
    return url, username, key, database


@pytest.fixture(scope="session")
def unique_name_prefix() -> str:
    # Unique prefix to avoid resource name collisions across parallel runs
    return f"pytest_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def vcr_config():
    """Configure VCR via pytest-recording's vcr_config fixture.

    - Redact Authorization header
    - Decode compressed responses for readability
    - Sanitize request URIs (mask host) and response bodies (drop non-test DBs, mask IPs)
    """
    import os
    import json
    import re
    from urllib.parse import urlparse, urlunparse

    test_db = os.getenv("TCVECTORDB_DATABASE", "<TEST_DB>")
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    def _redact_request(request):  # vcrpy hook
        try:
            parsed = urlparse(request.uri)
            # sanitize query params but keep them for disambiguation
            query_items = []
            if parsed.query:
                for q in parsed.query.split("&"):
                    if not q:
                        continue
                    if "=" in q:
                        k, v = q.split("=", 1)
                    else:
                        k, v = q, ""
                    lk = k.lower()
                    if any(s in lk for s in ("token", "key", "password", "auth")):
                        v = "<REDACTED>"
                    if lk in ("database", "db"):
                        v = test_db
                    v = ip_re.sub("<REDACTED_IP>", v)
                    query_items.append(f"{k}={v}")
            new_query = "&".join(query_items) if query_items else ""
            # replace host with placeholder, keep scheme/path/query
            new_uri = urlunparse(
                (parsed.scheme or "http", "vdb.local", parsed.path, "", new_query, "")
            )
            request.uri = ip_re.sub("vdb.local", new_uri)
            # sanitize JSON body but keep structure/params
            if hasattr(request, "body") and request.body:
                body = request.body
                if isinstance(body, (bytes, bytearray)):
                    body_str = body.decode("utf-8", errors="ignore")
                else:
                    body_str = str(body)
                try:
                    data = json.loads(body_str)
                    # standardize database field
                    if isinstance(data, dict):
                        if "database" in data:
                            data["database"] = test_db
                        # also patch nested search/query payloads
                        if "search" in data and isinstance(data["search"], dict):
                            if "documentIds" in data["search"]:
                                pass  # keep ids
                        if "query" in data and isinstance(data["query"], dict):
                            pass  # keep filters/sort/ids
                    request.body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                except Exception:
                    # plain text bodies: just mask IPs
                    request.body = ip_re.sub("<REDACTED_IP>", body_str).encode("utf-8")
        except Exception:
            pass
        return request

    def _redact_response(response):  # vcrpy hook
        try:
            body = response.get("body", {})
            s = body.get("string")
            if isinstance(s, (bytes, bytearray)):
                s = s.decode("utf-8", errors="ignore")
            if not isinstance(s, str) or not s:
                return response
            # mask IPs
            s_masked = ip_re.sub("<REDACTED_IP>", s)
            # attempt JSON sanitization
            try:
                data = json.loads(s_masked)
                # limit databases listing to test db only
                if isinstance(data, dict) and "databases" in data:
                    data["databases"] = [test_db]
                    info = data.get("info", {})
                    if isinstance(info, dict):
                        data["info"] = {test_db: info.get(test_db, {})}
                    s_masked = json.dumps(data, ensure_ascii=False)
            except Exception:
                pass
            # ensure body bytes for cassette
            response["body"]["string"] = s_masked.encode("utf-8")
        except Exception:
            pass
        return response

    return {
        "filter_headers": ["Authorization"],
        "decode_compressed_response": True,
        "before_record_request": _redact_request,
        "before_record_response": _redact_response,
    }


@pytest_asyncio.fixture()
async def db_client():
    url, username, key, database = _load_env()
    client = AsyncVectorDBClient(
        url=url,
        username=username,
        key=key,
        timeout=30,
        read_consistency=ReadConsistency.STRONG_CONSISTENCY,
    )
    # Attach default database name for convenience in tests/fixtures
    setattr(client, "default_db", database)
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture()
async def ensure_database(db_client: AsyncVectorDBClient):
    await db_client.create_database_if_not_exists(db_client.default_db)  # idempotent
    return db_client.default_db


@pytest_asyncio.fixture()
async def temp_collection(
    request,
    db_client: AsyncVectorDBClient,
    ensure_database: str,
    unique_name_prefix: str,
):
    # Use test-specific name to avoid cross-test reuse
    coll = f"{unique_name_prefix}_{request.node.name}_coll"
    index = Index()
    index.add(
        VectorIndex(
            name="vector",
            dimension=3,
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params={"M": 8, "efConstruction": 80},
        )
    )
    index.add(
        FilterIndex(
            name="id",
            field_type=FieldType.String,
            index_type=IndexType.PRIMARY_KEY,
        )
    )
    index.add(
        FilterIndex(
            name="tag",
            field_type=FieldType.String,
            index_type=IndexType.FILTER,
        )
    )
    index.add(
        FilterIndex(
            name="page",
            field_type=FieldType.Uint64,
            index_type=IndexType.FILTER,
        )
    )
    await db_client.create_collection_if_not_exists(
        database_name=db_client.default_db,
        collection_name=coll,
        shard=1,
        replicas=1,
        index=index,
        timeout=30,
    )
    try:
        yield coll
    finally:
        try:
            await db_client.drop_collection(db_client.default_db, coll)
        except Exception:
            pass


def pytest_collection_modifyitems(config, items):
    # Automatically apply VCR to all tests, unless explicitly opted out via marker `novcr`.
    for item in items:
        if item.get_closest_marker("novcr"):
            continue
        item.add_marker(pytest.mark.vcr)


@pytest_asyncio.fixture()
async def temp_embedding_collection(
    request,
    db_client: AsyncVectorDBClient,
    ensure_database: str,
    unique_name_prefix: str,
):
    coll = f"{unique_name_prefix}_{request.node.name}_embed_coll"

    # Create collection with embedding config
    index = Index()
    index.add(
        VectorIndex(
            name="vector",
            dimension=768,
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params={"M": 16, "efConstruction": 200},
        )
    )
    index.add(
        FilterIndex(
            name="id",
            field_type=FieldType.String,
            index_type=IndexType.PRIMARY_KEY,
        )
    )
    embedding = Embedding(vector_field="vector", field="text", model_name="bge-base-zh")
    await db_client.create_collection_if_not_exists(
        database_name=db_client.default_db,
        collection_name=coll,
        shard=1,
        replicas=1,
        index=index,
        embedding=embedding,
        timeout=60,
    )
    try:
        yield coll
    finally:
        try:
            await db_client.drop_collection(db_client.default_db, coll)
        except Exception:
            pass
