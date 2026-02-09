from rest_framework import serializers
from .models import Thread, Message


class MessageSerializer(serializers.ModelSerializer):
    type = serializers.CharField(source="role", read_only=True)

    class Meta:
        model = Message
        fields = [
            "id",
            "type",
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
            "created_at",
            "metadata",
        ]
        read_only_fields = ["id", "created_at"]


class ThreadSerializer(serializers.ModelSerializer):
    thread_id = serializers.UUIDField(source="id", read_only=True)

    class Meta:
        model = Thread
        fields = ["thread_id", "title", "created_at", "updated_at", "metadata"]
        read_only_fields = ["thread_id", "created_at", "updated_at"]


class ThreadDetailSerializer(ThreadSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ThreadSerializer.Meta):
        fields = ThreadSerializer.Meta.fields + ["messages"]
