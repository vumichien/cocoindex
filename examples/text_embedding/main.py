from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from typing import Any
import cocoindex
import os
from numpy.typing import NDArray
import numpy as np
from datetime import timedelta


@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[NDArray[np.float32]]:
    """
    Embed the text using a SentenceTransformer model.
    This is a shared logic between indexing and querying, so extract it as a function."""
    # You can also switch to remote embedding model:
    #   return text.transform(
    #       cocoindex.functions.EmbedText(
    #           api_type=cocoindex.LlmApiType.OPENAI,
    #           model="text-embedding-3-small",
    #       )
    #   )
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


@cocoindex.flow_def(name="TextEmbedding")
def text_embedding_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    """
    Define an example flow that embeds text into a vector database.
    """
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="markdown_files"),
        refresh_interval=timedelta(seconds=5),
    )

    doc_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=2000,
            chunk_overlap=500,
        )

        with doc["chunks"].row() as chunk:
            chunk["embedding"] = text_to_embedding(chunk["text"])
            doc_embeddings.collect(
                filename=doc["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )

    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


def search(pool: ConnectionPool, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    # Get the table name, for the export target in the text_embedding_flow above.
    table_name = cocoindex.utils.get_target_default_name(
        text_embedding_flow, "doc_embeddings"
    )
    # Evaluate the transform flow defined above with the input query, to get the embedding.
    query_vector = text_to_embedding.eval(query)
    # Run the query and get the results.
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT filename, text, embedding <=> %s AS distance
                FROM {table_name} ORDER BY distance LIMIT %s
            """,
                (query_vector, top_k),
            )
            return [
                {"filename": row[0], "text": row[1], "score": 1.0 - row[2]}
                for row in cur.fetchall()
            ]


def _main() -> None:
    # Initialize the database connection pool.
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    # Run queries in a loop to demonstrate the query capabilities.
    while True:
        query = input("Enter search query (or Enter to quit): ")
        if query == "":
            break
        # Run the query function with the database connection pool and the query.
        results = search(pool, query)
        print("\nSearch results:")
        for result in results:
            print(f"[{result['score']:.3f}] {result['filename']}")
            print(f"    {result['text']}")
            print("---")
        print()


if __name__ == "__main__":
    load_dotenv()
    cocoindex.init()
    _main()
