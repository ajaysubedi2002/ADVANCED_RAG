from __future__ import annotations

from typing import List

from app.retrieval.parent_expansion import ExpandedParent
from app.runtime_settings import settings

# Prompt templates

SYSTEM_PROMPT = "You are an assistant. Answer the question using only the text in the CONTEXT below. Do not add anything not in the context."

CONTEXT_SECTION_TEMPLATE = "[{index}] {text}"

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# Builder
class PromptBuilder:
    """
    Assembles the final prompt sent to the LLM from parent contexts.

    Parameters
    ----------
    max_context_chars:  Hard cap on total context characters to avoid
                        exceeding the model's context window.
    """

    def __init__(self, max_context_chars: int | None = None) -> None:
        self.max_context_chars = max_context_chars or settings.MAX_RAG_CONTEXT_CHARS

    def build(
        self,
        question: str,
        expanded_parents: List[ExpandedParent],
    ) -> tuple[str, str]:
        """
        Build the (system_prompt, user_prompt) tuple for the LLM.

        Context sections are included in order of ``best_child_score`` until
        the ``max_context_chars`` budget is consumed.

        Returns
        -------
        (system_prompt: str, user_prompt: str)
        """
        context_sections: List[str] = []
        total_chars = 0

        for i, parent in enumerate(expanded_parents, start=1):
            section = CONTEXT_SECTION_TEMPLATE.format(
                index=i,
                document_id=parent.document_id,
                text=parent.text.strip(),
            )
            if total_chars + len(section) > self.max_context_chars:
                # Truncate the current section to fit
                remaining = self.max_context_chars - total_chars
                if remaining > 200:  # only add if there's meaningful space
                    section = section[:remaining] + "\n[truncated]"
                    context_sections.append(section)
                break
            context_sections.append(section)
            total_chars += len(section)

        context_block = "\n\n".join(context_sections) if context_sections else settings.RAG_CONTEXT_PLACEHOLDER

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context_block,
            question=question.strip(),
        )

        return SYSTEM_PROMPT, user_prompt
