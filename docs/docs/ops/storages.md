---
title: Storages
description: CocoIndex Built-in Storages
---

# CocoIndex Built-in Storages

## Postgres

`Postgres` exports data to Postgres database (with pgvector extension).

The spec takes the following fields:

*   `database_url` (type: `str`, optional): The URL of the Postgres database to use as the internal storage, e.g. `postgres://cocoindex:cocoindex@localhost/cocoindex`. If unspecified, will use the same database as the [internal storage](/docs/core/basics#internal-storage).

*   `table_name` (type: `str`, optional): The name of the table to store to. If unspecified, will generate a new automatically. We recommend specifying a name explicitly if you want to directly query the table. It can be omitted if you want to use CocoIndex's query handlers to query the table.
