from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):

        self.documents = []
        self.bm25 = None

    def build_index(self, chunks):

        self.documents = chunks

        tokenized_documents = [
            chunk.text.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query,
        top_k=20
    ):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            chunk = self.documents[index]

            results.append({
                "chunk_id": chunk.id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "score": float(scores[index])
            })

        return results