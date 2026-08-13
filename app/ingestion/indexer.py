from sentence_transformers import SentenceTransformer
from config import settings

class EmbeddingModel:

    def __init__(
        self,
        model_name = settings.EMBED_MODEL
    ):

        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

    def embed_query(self, query):

        return self.model.encode(
            query,
            normalize_embeddings=True
        )