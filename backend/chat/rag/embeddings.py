import os
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model():
    """Get the OpenAI embeddings model."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    model = get_embeddings_model()
    return model.embed_query(text)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    model = get_embeddings_model()
    return model.embed_documents(texts)
