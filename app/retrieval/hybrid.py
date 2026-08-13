class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_retriever
    ):

        self.vector_store = vector_store
        self.bm25 = bm25_retriever

    def rrf(
        self,
        rankings,
        k=60
    ):

        scores = {}

        for ranking in rankings:

            for rank, item in enumerate(
                ranking,
                start=1
            ):

                chunk_id = item["chunk_id"]

                scores.setdefault(
                    chunk_id,
                    {
                        "score": 0,
                        "item": item
                    }
                )

                scores[chunk_id]["score"] += (
                    1 / (k + rank)
                )

        results = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return results

    def search(
        self,
        query_vector,
        query,
        top_k=10
    ):

        vector_results = (
            self.vector_store.search(
                query_vector,
                limit=20
            )
        )

        vector_ranking = []

        for result in vector_results:

            vector_ranking.append({
                "chunk_id":
                    result.payload["chunk_id"],

                "parent_id":
                    result.payload["parent_id"],

                "text":
                    result.payload["text"]
            })

        bm25_results = self.bm25.search(
            query,
            top_k=20
        )

        combined = self.rrf(
            [
                vector_ranking,
                bm25_results
            ]
        )

        return combined[:top_k]