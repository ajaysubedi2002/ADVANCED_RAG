import json
import uuid
from pathlib import Path

from drf_spectacular.utils import OpenApiExample, OpenApiResponse, extend_schema
from rest_framework.decorators import api_view
from rest_framework.response import Response

from app.ingestion.chunker import create_parent_child_chunks
from app.ingestion.indexer import EmbeddingModel
from app.ingestion.parser import load_document
from app.retrieval.vector_search import VectorStore


_embedding_model = EmbeddingModel()
_vector_store = VectorStore()


@extend_schema(
	tags=["Health"],
	description="Health check endpoint.",
	responses={200: OpenApiResponse(description="Service is healthy.")},
)
@api_view(["GET"])
def api_root(request):
	return Response({"status": "ok"})


@extend_schema(
	tags=["Ingestion"],
	description="Ingest a local PDF or TXT document into the vector store.",
	request={
		"application/json": {
			"type": "object",
			"properties": {
				"file_path": {"type": "string", "description": "Absolute path to the document on the server."},
				"document_id": {"type": "string", "description": "Optional custom document id."},
				"parent_size": {"type": "integer", "default": 1000},
				"child_size": {"type": "integer", "default": 300},
			},
			"required": ["file_path"],
		}
	},
	responses={
		201: OpenApiResponse(description="Document ingested successfully."),
		400: OpenApiResponse(description="Invalid request body or unsupported file."),
		404: OpenApiResponse(description="Provided file_path does not exist."),
		500: OpenApiResponse(description="Unexpected ingestion failure."),
	},
	examples=[
		OpenApiExample(
			"Ingest PDF",
			value={
				"file_path": "/drive_d/Advanced_Rag/data/sample.pdf",
				"document_id": "doc-001",
				"parent_size": 1000,
				"child_size": 300,
			},
			request_only=True,
		),
	],
)
@api_view(["POST"])
def ingest_document(request):
	try:
		payload = request.data
		if isinstance(payload, str):
			payload = json.loads(payload or "{}")
		if payload is None:
			payload = {}
	except json.JSONDecodeError:
		return Response({"error": "Invalid JSON body"}, status=400)
	except Exception:
		return Response({"error": "Invalid request body"}, status=400)

	file_path = payload.get("file_path")
	if not file_path:
		return Response({"error": "file_path is required"}, status=400)

	document_id = payload.get("document_id") or str(uuid.uuid4())
	try:
		parent_size = int(payload.get("parent_size", 1000))
		child_size = int(payload.get("child_size", 300))
	except (TypeError, ValueError):
		return Response({"error": "parent_size and child_size must be integers"}, status=400)

	path = Path(file_path).expanduser().resolve()
	if not path.exists() or not path.is_file():
		return Response({"error": f"File not found: {path}"}, status=404)

	try:
		parsed_doc = load_document(str(path))
		parents, children = create_parent_child_chunks(
			document_id=document_id,
			parsed_doc=parsed_doc,
			parent_size=parent_size,
			child_size=child_size,
		)

		if not children:
			return Response(
				{
					"status": "ok",
					"document_id": document_id,
					"message": "No child chunks generated; nothing indexed.",
					"parent_chunks": 0,
					"child_chunks": 0,
				}
			)

		embeddings = _embedding_model.embed_documents([child.text for child in children])
		_vector_store.create_collection(vector_size=embeddings.shape[1])
		_vector_store.add_chunks(children, embeddings)

		return Response(
			{
				"status": "ok",
				"document_id": document_id,
				"source": str(path),
				"parent_chunks": len(parents),
				"child_chunks": len(children),
			},
			status=201,
		)
	except ValueError as exc:
		return Response({"error": str(exc)}, status=400)
	except Exception as exc:
		return Response({"error": f"Ingestion failed: {exc}"}, status=500)
