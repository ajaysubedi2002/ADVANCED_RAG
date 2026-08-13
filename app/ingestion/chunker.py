import uuid
from typing import Any

from app.models.schemas import (ParentChunk,ChildChunk)


def split_text(text: str,chunk_size: int = 300,overlap: int = 50):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def create_parent_child_chunks(document_id: str, parsed_doc: Any, parent_size=1000, child_size=300):
    parents, children = [], []

    for parent_group in parsed_doc.chunks(max_chars=parent_size):
        parent_id = str(uuid.uuid4())
        parents.append(ParentChunk(
            id=parent_id,
            document_id=document_id,
            text=parent_group["text"],
            metadata={"pages": parent_group["pages"]},
        ))

        for child_text in split_text(parent_group["text"], chunk_size=child_size, overlap=50):
            children.append(ChildChunk(
                id=str(uuid.uuid4()),
                parent_id=parent_id,
                document_id=document_id,
                text=child_text,
                metadata={"pages": parent_group["pages"]},
            ))

    return parents, children