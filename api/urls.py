from django.urls import path

from .views import api_root, ingest_document


urlpatterns = [
	path("", api_root, name="api-root"),
	path("documents/ingest/", ingest_document, name="ingest-document"),
]
