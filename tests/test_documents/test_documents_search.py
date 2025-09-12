from aiotcvectordb.model import Document, Filter


async def test_search_by_id_returns_self_top1(db_client, temp_collection):
    docs = [
        Document(id="s1", vector=[0.9, 0.0, 0.0]),
        Document(id="s2", vector=[0.0, 0.9, 0.0]),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    # Minimal param object compatible with vendor's `vars(params)` usage
    params = type("HNSW", (), {"ef": 100})()
    res = await db_client.search_by_id(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        document_ids=["s1"],
        params=params,
        retrieve_vector=False,
        limit=1,
        timeout=30,
    )
    assert isinstance(res, list)
    assert res and isinstance(res[0], list)
    assert res[0] and res[0][0].get("id") == "s1"


async def test_search_vectors_shape(db_client, temp_collection):
    docs = [
        Document(id="v1", vector=[0.1, 0.2, 0.3]),
        Document(id="v2", vector=[0.1, 0.2, 0.31]),
        Document(id="v3", vector=[0.9, 0.0, 0.0]),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    res = await db_client.search(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        vectors=[[0.1, 0.2, 0.3], [0.9, 0.0, 0.0]],
        limit=2,
        retrieve_vector=False,
        timeout=30,
    )
    assert isinstance(res, list)
    assert len(res) == 2
    assert all(isinstance(group, list) for group in res)
    # For the first query vector ~ [0.1,0.2,0.3], top-1 should be among v1/v2 (closer ones)
    first_top = res[0][0] if res[0] else {}
    assert first_top.get("id") in {"v1", "v2"}


async def test_search_with_filter_and_output_fields(db_client, temp_collection):
    # Prepare docs with tags for filtering
    docs = [
        Document(id="fa", vector=[0.11, 0.22, 0.33], tag="a", page=1),
        Document(id="fb", vector=[0.12, 0.21, 0.31], tag="b", page=2),
        Document(id="fc", vector=[0.13, 0.20, 0.30], tag="a", page=3),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    res = await db_client.search(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        vectors=[[0.11, 0.22, 0.33]],
        filter=Filter('tag="a"'),
        limit=5,
        retrieve_vector=False,
        output_fields=["id", "tag"],
        timeout=30,
    )
    assert isinstance(res, list)
    assert res and isinstance(res[0], list)
    for rec in res[0]:
        assert rec.get("tag") == "a"
        # Only requested fields should be present (id, tag). Some systems may also return internal fields like score.
        # We at least ensure a non-requested field like 'page' is absent.
        assert "page" not in rec
