# Ingestion package.
# Import directly from submodules to avoid triggering heavy dependencies
# (rank_bm25, chromadb, ollama) at Django URL-load time.
#
# Good:  from app.ingestion.exceptions import DocumentParsingError
# Avoid: from app.ingestion import DocumentParsingError
