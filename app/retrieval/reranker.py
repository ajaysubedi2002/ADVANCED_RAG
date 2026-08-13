from sentence_transformers import CrossEncoder
from config import settings 


class Reranker:

    def __init__(
        self,
        model_name= settings.RERANK_MODEL
    ):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        pairs = [
            (
                query,
                document["item"]["text"]
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = []

        for document, score in zip(
            documents,
            scores
        ):

            document["rerank_score"] = float(
                score
            )

            ranked.append(document)

        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]