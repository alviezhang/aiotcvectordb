from aiotcvectordb.model import Document


async def test_search_by_text_with_embedding_collection(
    db_client, temp_embedding_collection
):
    # Distinct Chinese texts for clearer semantic separation
    docs = [
        Document(id="t_apple", text="苹果很好吃"),
        Document(id="t_banana", text="香蕉很好吃"),
    ]
    res = await db_client.upsert(
        database_name=db_client.default_db,
        collection_name=temp_embedding_collection,
        documents=docs,
        build_index=True,
        timeout=60,
    )
    assert isinstance(res, dict)
    out = await db_client.search_by_text(
        database_name=db_client.default_db,
        collection_name=temp_embedding_collection,
        embedding_items=["苹果"],
        retrieve_vector=False,
        limit=1,
        timeout=60,
    )
    # top-1 should be the apple text
    # search_by_text returns a dict with key "documents" in async model wrapper
    docs_res = out.get("documents") if isinstance(out, dict) else out
    assert isinstance(docs_res, list)
    assert docs_res and isinstance(docs_res[0], list)
    assert docs_res[0] and docs_res[0][0].get("id") == "t_apple"
