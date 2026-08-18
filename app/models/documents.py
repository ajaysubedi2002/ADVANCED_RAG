from __future__ import annotations

from django.db import models


class Document(models.Model):
    """Top-level document record."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.CharField(primary_key=True, max_length=64)
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=20)
    file_path = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "api"
        db_table = "documents"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.filename} ({self.id})"


class ParentChunk(models.Model):
    """
    Larger context window chunk.

    Stored in PostgreSQL for parent expansion after reranking.
    NOT embedded into ChromaDB – only child chunks carry embeddings.
    """

    id = models.CharField(primary_key=True, max_length=64)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="parent_chunks",
        db_index=True,
    )
    text = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "api"
        db_table = "parent_chunks"
        indexes = [
            models.Index(fields=["document"]),
        ]

    def __str__(self) -> str:
        return self.id


class ChildChunk(models.Model):
    """
    Smaller retrieval unit chunk.

    Indexed in both ChromaDB (dense embedding) and BM25 (sparse keyword).
    """

    id = models.CharField(primary_key=True, max_length=64)
    parent = models.ForeignKey(
        ParentChunk,
        on_delete=models.CASCADE,
        related_name="children",
        db_index=True,
    )
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="child_chunks",
        db_index=True,
    )
    text = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "api"
        db_table = "child_chunks"
        indexes = [
            models.Index(fields=["document"]),
            models.Index(fields=["parent"]),
        ]

    def __str__(self) -> str:
        return self.id
