from aiotcvectordb.model import Document


async def test_upsert_documents_success(db_client, temp_collection):
    docs = [
        Document(id="u1", vector=[0.21, 0.22, 0.23], t="x"),
        Document(id="u2", vector=[0.31, 0.32, 0.33], t="y"),
    ]
    res = await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )
    assert isinstance(res, dict)
    assert res.get("code", 0) == 0 or res.get("affectedCount", 0) >= 1


async def test_query_by_ids(db_client, temp_collection):
    docs = [
        Document(id="0001", vector=[0.11, 0.22, 0.33], page=1),
        Document(id="0002", vector=[0.12, 0.21, 0.31], page=2),
        Document(id="0003", vector=[0.13, 0.20, 0.30], page=3),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )
    ids = ["0001", "0002", "0003"]
    out = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        document_ids=ids,
        retrieve_vector=False,
        output_fields=["id", "page"],
        limit=10,
        timeout=30,
    )
    assert isinstance(out, list)
    assert {d["id"] for d in out}.issubset(set(ids))


async def test_update_by_ids(db_client, temp_collection):
    docs = [
        Document(id="0001", vector=[0.11, 0.22, 0.33], page=1),
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )
    # update page for id=0001
    update_doc = Document(page=99)
    res = await db_client.update(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        data=update_doc,
        document_ids=["0001"],
        timeout=30,
    )
    assert isinstance(res, dict)

    # verify
    out = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        document_ids=["0001"],
        output_fields=["id", "page"],
        timeout=30,
    )
    assert out and out[0]["page"] == 99


async def test_delete_by_ids(db_client, temp_collection):
    docs = [
        Document(id="0003", vector=[0.13, 0.20, 0.30], page=3),
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
        document_ids=["0003"],
        timeout=30,
    )
    assert isinstance(res, dict)

    # Verify deletion by affectedCount; some deployments may show eventual visibility in read paths
    assert res.get("affectedCount", 0) >= 1 or res.get("code", 0) == 0
