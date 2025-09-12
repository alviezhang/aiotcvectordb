from aiotcvectordb.model import Document, Filter


async def test_query_sort_ascending(db_client, temp_collection):
    docs = [
        Document(id="q1", vector=[0.11, 0.22, 0.33], tag="a", page=3),
        Document(id="q2", vector=[0.12, 0.21, 0.31], tag="a", page=1),
        Document(id="q3", vector=[0.13, 0.20, 0.30], tag="a", page=2),
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
        output_fields=["id", "page"],
        sort={"fieldName": "page", "direction": "asc"},
        limit=10,
        timeout=30,
    )
    pages = [d["page"] for d in out]
    assert pages == sorted(pages)


async def test_query_pagination_and_field_trimming(db_client, temp_collection):
    docs = [
        Document(id=f"p{i}", vector=[0.11, 0.22, 0.33], tag="p", page=i)
        for i in range(1, 6)
    ]
    await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        documents=docs,
        build_index=True,
        timeout=30,
    )

    # First page (limit=2, offset=0)
    page1 = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="p"'),
        output_fields=["id"],
        sort={"fieldName": "page", "direction": "asc"},
        limit=2,
        offset=0,
        timeout=30,
    )
    # Second page (offset=2)
    page2 = await db_client.query(
        database_name=db_client.default_db,
        collection_name=temp_collection,
        filter=Filter('tag="p"'),
        output_fields=["id"],
        sort={"fieldName": "page", "direction": "asc"},
        limit=2,
        offset=2,
        timeout=30,
    )

    assert len(page1) == 2 and len(page2) == 2
    # Ensure IDs differ between pages
    assert set(d["id"] for d in page1).isdisjoint(set(d["id"] for d in page2))
    # Ensure only requested field present
    assert all(list(doc.keys()) == ["id"] for doc in page1 + page2)
