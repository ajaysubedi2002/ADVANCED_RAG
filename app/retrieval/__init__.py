# Retrieval package.
# Import directly from submodules to avoid triggering chromadb / sentence-transformers
# at Django URL-load time.
#
# Good:  from app.retrieval.dense_search import VectorStore
# Avoid: from app.retrieval import VectorStore
