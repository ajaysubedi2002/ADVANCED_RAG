from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.runtime_settings import settings
from app.models.documents import ParentChunk as ParentORM 


logger = logging.getLogger(__name__)


# Result schema

@dataclass
class ExpandedParent:
    parent_id: str
    document_id: str
    text: str
    child_ids: List[str] = field(default_factory=list)   # children that triggered this parent
    best_child_score: float = 0.0                        # score of the highest-ranked child
    metadata: Dict[str, Any] = field(default_factory=dict)


# Expander

class ParentExpander:
    """
    Maps reranked child chunks to their parent documents.

    Parameters
    ----------
    top_k:           Maximum number of parent contexts to return.
    parent_store:    Optional in-memory dict ``{parent_id: ParentChunk}``
                     used as a fallback when the ORM is not available.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        parent_store: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.top_k = top_k if top_k is not None else settings.PARENT_CONTEXT_TOP_K
        self._parent_store = parent_store or {}

  
    # Lookup helpers
  

    def _lookup_parent_orm(self, parent_id: str) -> Optional[Any]:
        """Attempt to fetch the parent from Django ORM (avoids circular import)."""
        try:
            return ParentORM.objects.filter(id=parent_id).first()
        except Exception:
            return None

    def _lookup_parent(self, parent_id: str) -> Optional[Any]:
        """Return the parent object from memory store or ORM."""
        if parent_id in self._parent_store:
            return self._parent_store[parent_id]
        return self._lookup_parent_orm(parent_id)

  
    # Public API
    def expand(self, reranked_chunks: List[Any]) -> List[ExpandedParent]:
        """
        Expand a list of reranked child chunks into their parent contexts.

        Parameters
        ----------
        reranked_chunks:
            Iterable of objects with at least ``chunk_id``, ``parent_id``,
            ``document_id``, and ``score`` attributes (compatible with
            ``RerankedResult`` and ``FusedResult``).

        Returns
        -------
        Deduplicated list of ``ExpandedParent``, capped to ``self.top_k``,
        ordered by the best child score descending.
        """
        if not reranked_chunks:
            return []

        # Build parent_id → ExpandedParent, keeping best child score
        seen: Dict[str, ExpandedParent] = {}

        for chunk in reranked_chunks:
            parent_id = getattr(chunk, "parent_id", None)
            if not parent_id:
                logger.warning("Chunk %s has no parent_id; skipping expansion", getattr(chunk, "chunk_id", "?"))
                continue

            chunk_id = getattr(chunk, "chunk_id", "")
            score = float(getattr(chunk, "score", getattr(chunk, "rrf_score", 0.0)))

            if parent_id not in seen:
                parent = self._lookup_parent(parent_id)
                if parent is None:
                    logger.warning("Parent '%s' not found; using child text as fallback", parent_id)
                    parent_text = getattr(chunk, "text", "")
                    parent_doc_id = getattr(chunk, "document_id", "")
                    parent_meta: dict = {}
                else:
                    parent_text = getattr(parent, "text", "")
                    parent_doc_id = getattr(parent, "document_id", "") or getattr(parent, "document_id", "")
                    # ORM model stores document_id as FK; try both attribute forms
                    if not parent_doc_id:
                        parent_doc_id = str(getattr(parent, "document_id", ""))
                    parent_meta = getattr(parent, "metadata", {}) or {}

                seen[parent_id] = ExpandedParent(
                    parent_id=parent_id,
                    document_id=parent_doc_id,
                    text=parent_text,
                    child_ids=[chunk_id],
                    best_child_score=score,
                    metadata=parent_meta,
                )
            else:
                seen[parent_id].child_ids.append(chunk_id)
                if score > seen[parent_id].best_child_score:
                    seen[parent_id].best_child_score = score

        # Sort by best child score descending
        ordered = sorted(
            seen.values(),
            key=lambda p: p.best_child_score,
            reverse=True,
        )

        result = ordered[: self.top_k]

        logger.info(
            "Parent expansion complete | children_in=%d parents_out=%d top_k=%d",
            len(reranked_chunks),
            len(result),
            self.top_k,
        )
        return result
