from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import Document
from chat.rag.ingest import ingest_document


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser])
def document_list(request):
    if request.method == "GET":
        documents = Document.objects.all()
        data = [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "chunks": doc.chunk_index,
                "created_at": doc.created_at.isoformat(),
            }
            for doc in documents
        ]
        return Response(data)

    if request.method == "POST":
        file = request.FILES.get("file")
        if not file:
            return Response(
                {"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            doc, chunk_count = ingest_document(file, file.name)
            return Response(
                {
                    "document_id": str(doc.id),
                    "chunks": chunk_count,
                    "status": "indexed",
                },
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(["DELETE"])
def document_detail(request, document_id):
    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    doc.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)
