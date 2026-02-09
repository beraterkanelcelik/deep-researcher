from django.urls import path
from . import views
from . import views_stream

urlpatterns = [
    path("threads/", views.thread_list, name="thread-list"),
    path("threads/<uuid:thread_id>/", views.thread_detail, name="thread-detail"),
    path("threads/<uuid:thread_id>/state", views.thread_state, name="thread-state"),
    path(
        "threads/<uuid:thread_id>/runs/stream",
        views_stream.stream_run,
        name="thread-stream",
    ),
    path(
        "threads/<uuid:thread_id>/runs/resume",
        views_stream.resume_run,
        name="thread-resume",
    ),
]
