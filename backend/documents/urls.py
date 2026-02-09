from django.urls import path
from . import views

urlpatterns = [
    path("documents/upload", views.document_list, name="document-list"),
    path("documents/", views.document_list, name="document-list-get"),
    path(
        "documents/<uuid:document_id>/",
        views.document_detail,
        name="document-detail",
    ),
]
