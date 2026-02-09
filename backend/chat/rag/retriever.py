from documents.models import Embedding
from .embeddings import generate_embedding


def retrieve_documents(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve relevant document chunks using pgvector similarity search."""
    # Check if there are any embeddings
    if not Embedding.objects.exists():
        return []

    query_embedding = generate_embedding(query)

    # Use pgvector cosine distance ordering
    from pgvector.django import CosineDistance

    results = (
        Embedding.objects.annotate(distance=CosineDistance("embedding", query_embedding))
        .order_by("distance")[:top_k]
    )

    docs = []
    for result in results:
        docs.append(
            {
                "content": result.content,
                "filename": result.metadata.get("filename", "unknown"),
                "distance": float(result.distance),
                "document_id": str(result.document_id),
            }
        )

    return docs
