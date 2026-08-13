from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ChildChunk:
    id: str
    parent_id: str
    document_id: str

    text: str

    metadata: Dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    id: str
    document_id: str

    text: str

    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk_id: str
    parent_id: str
    text: str

    score: float = 0.0

    metadata: Dict = field(default_factory=dict)
    