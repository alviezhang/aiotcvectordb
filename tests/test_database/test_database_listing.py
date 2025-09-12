async def test_list_databases_includes_configured_db(db_client):
    await db_client.create_database_if_not_exists(db_client.default_db)
    dbs = await db_client.list_databases()
    names = [db.database_name for db in dbs]
    assert db_client.default_db in names
