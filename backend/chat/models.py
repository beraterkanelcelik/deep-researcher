import uuid
from django.db import models


class Thread(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, default="", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "threads"
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or str(self.id)


class Message(models.Model):
    ROLE_CHOICES = [
        ("human", "Human"),
        ("ai", "AI"),
        ("tool", "Tool"),
        ("system", "System"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    tool_calls = models.JSONField(default=list, blank=True)
    tool_call_id = models.CharField(max_length=255, blank=True, default="")
    name = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "messages"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
        ]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class ResearchReport(models.Model):
    """Persisted research report produced by the deep research pipeline."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    thread = models.ForeignKey(
        Thread, on_delete=models.CASCADE, related_name="research_reports"
    )
    title = models.CharField(max_length=500)
    summary = models.TextField(default="")
    key_findings = models.JSONField(default=list, blank=True)
    sources = models.JSONField(default=list, blank=True)
    tags = models.JSONField(default=list, blank=True)
    methodology = models.TextField(default="", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_reports"
        ordering = ["-created_at"]

    def __str__(self):
        return self.title
