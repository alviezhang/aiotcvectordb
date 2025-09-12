from aiotcvectordb.model import Document, Filter


async def test_query_with_filter(db_client, temp_collection):
    docs = [
        Document(id="f1", vector=[0.11, 0.22, 0.33], tag="a", page=1),
        Document(id="f2", vector=[0.12, 0.21, 0.31], tag="b", page=2),
        Document(id="f3", vector=[0.13, 0.20, 0.30], tag="a", page=3),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    out = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="a"'),
        output_fields=["id", "tag"],
        limit=10,
        timeout=30,
    )
    assert isinstance(out, list)
    assert out and all(doc["tag"] == "a" for doc in out)


async def test_update_with_filter(db_client, temp_collection):
    docs = [
        Document(id="u1", vector=[0.21, 0.22, 0.23], tag="a", page=1),
        Document(id="u2", vector=[0.31, 0.32, 0.33], tag="a", page=2),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    from aiotcvectordb.model import Document as UpdateDoc

    res = await db_client.update(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        data=UpdateDoc(page=100),
        filter=Filter('tag="a"'),
        timeout=30,
    )
    assert isinstance(res, dict)

    out = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="a"'),
        output_fields=["page"],
        limit=10,
        timeout=30,
    )
    assert all(doc.get("page") == 100 for doc in out)


async def test_delete_with_filter_limit(db_client, temp_collection):
    docs = [
        Document(id="d1", vector=[0.11, 0.22, 0.33], tag="a", page=1),
        Document(id="d2", vector=[0.12, 0.21, 0.31], tag="a", page=2),
        Document(id="d3", vector=[0.13, 0.20, 0.30], tag="b", page=3),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    res = await db_client.delete(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="a"'),
        limit=1,
        timeout=30,
    )
    assert isinstance(res, dict)

    out = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="a"'),
        output_fields=["id"],
        limit=10,
        timeout=30,
    )
    # Should be <= 1 remaining with tag="a"
    assert len(out) <= 1
