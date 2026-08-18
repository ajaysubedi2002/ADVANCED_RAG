from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    document_id: str
    filename: str
    file_type: str
    file_path: str
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_db_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "status": self.status,
            "metadata": self.metadata,
        }
