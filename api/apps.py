import logging
import os

from django.apps import AppConfig
from app.runtime_settings import validate_runtime_settings 

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    name = "api"

    def ready(self) -> None:
        """
        Called once when Django starts up (after all models are loaded).

        1. Validates runtime settings (chunk sizes, required env vars).
        2. Configures LangSmith tracing by pushing the four required env vars
           that LangChain reads automatically on first import:
               LANGCHAIN_TRACING_V2
               LANGCHAIN_API_KEY
               LANGCHAIN_PROJECT
               LANGCHAIN_ENDPOINT
           We set them here from our own LANGSMITH_* settings so the .env
           file stays clean and a single LANGSMITH_TRACING=false disables
           everything without touching any other file.
        """

        s = validate_runtime_settings()

        # LangSmith bootstrap
        _placeholder = s.LANGSMITH_API_KEY.startswith("your_") or not s.LANGSMITH_API_KEY
        if s.LANGSMITH_TRACING and not _placeholder:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = s.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = s.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = s.LANGSMITH_ENDPOINT
            logger.info(
                "LangSmith tracing enabled | project=%s endpoint=%s",
                s.LANGSMITH_PROJECT,
                s.LANGSMITH_ENDPOINT,
            )
        else:
            # Ensure it's explicitly off so no accidental tracing occurs
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            logger.info(
                "LangSmith tracing disabled "
                "(set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"
            )
