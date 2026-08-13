import chromadb
from chromadb.config import Settings
from config import settings as app_settings


class _Result:
    def __init__(self, payload: dict):
        self.payload = payload


class VectorStore:

    def __init__(
        self,
        collection_name=None,
        path=None
    ):
        self.collection_name = (
            collection_name or app_settings.COLLECTIONS_NAME
        )

        db_path = path or app_settings.CHROMA_DB_PATH

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = None

 

    def _get_collection(self):
        if self._collection is None:
            self._collection = (
                self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            )
        return self._collection


    def create_collection(self, vector_size=None):
        """Ensures the collection exists.  vector_size is accepted for
        API compatibility but ChromaDB infers dimensions automatically."""
        self._get_collection()

    def add_chunks(self, chunks, embeddings):

        ids = []
        vectors = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):
            ids.append(str(chunk.id))
            vectors.append(embedding.tolist())
            documents.append(chunk.text)
            metadatas.append({
                "chunk_id": chunk.id,
                "parent_id": chunk.parent_id,
                "document_id": chunk.document_id,
                "metadata": str(chunk.metadata)
            })

        self._get_collection().upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query_vector, limit=20):
        """Returns a list of _Result objects whose .payload dict keeps
        the same interface so the rest of the pipeline (hybrid.py etc.)
        requires no changes."""

        results = self._get_collection().query(
            query_embeddings=[query_vector.tolist()],
            n_results=limit,
            include=["documents", "metadatas"]
        )

        output = []

        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):
            output.append(
                _Result(payload={
                    "chunk_id": meta["chunk_id"],
                    "parent_id": meta["parent_id"],
                    "document_id": meta["document_id"],
                    "text": doc,
                    "metadata": meta.get("metadata", {})
                })
            )

        return output
