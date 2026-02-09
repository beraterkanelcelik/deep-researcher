import tiktoken
from documents.models import Document, Embedding
from .embeddings import generate_embeddings


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into chunks based on token count."""
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start = end - chunk_overlap

    return chunks


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file."""
    from docx import Document as DocxDocument

    doc = DocxDocument(file)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_txt(file) -> str:
    """Extract text from a plain text file."""
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return content


def extract_text(file, filename: str) -> str:
    """Extract text from file based on extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext in ("txt", "md", "csv", "json", "py", "js", "ts", "html", "css"):
        return extract_text_from_txt(file)
    else:
        return extract_text_from_txt(file)


def ingest_document(file, filename: str) -> tuple[Document, int]:
    """Ingest a document: extract text, chunk, embed, and store."""
    # Extract text
    text = extract_text(file, filename)

    # Create document record
    doc = Document.objects.create(
        filename=filename,
        content=text,
    )

    # Chunk
    chunks = chunk_text(text)

    if not chunks:
        return doc, 0

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store chunks with embeddings
    embedding_objects = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        embedding_objects.append(
            Embedding(
                document=doc,
                content=chunk,
                embedding=embedding,
                metadata={"chunk_index": i, "filename": filename},
            )
        )

    Embedding.objects.bulk_create(embedding_objects)

    # Update document chunk_index with total count
    doc.chunk_index = len(chunks)
    doc.save()

    return doc, len(chunks)
