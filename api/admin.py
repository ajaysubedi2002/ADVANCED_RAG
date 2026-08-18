from django.contrib import admin

from app.models.documents import ChildChunk, Document, ParentChunk


class ParentChunkInline(admin.TabularInline):
    model = ParentChunk
    extra = 0
    readonly_fields = ("id", "created_at")
    show_change_link = True


class ChildChunkInline(admin.TabularInline):
    model = ChildChunk
    extra = 0
    readonly_fields = ("id", "created_at")
    show_change_link = True


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("filename", "id", "file_type", "status", "created_at", "updated_at")
    list_filter = ("status", "file_type")
    search_fields = ("filename", "id")
    readonly_fields = ("id", "created_at", "updated_at")
    ordering = ("-created_at",)
    inlines = [ParentChunkInline]


@admin.register(ParentChunk)
class ParentChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "created_at")
    list_filter = ("document",)
    search_fields = ("id", "document__filename")
    readonly_fields = ("id", "created_at")
    raw_id_fields = ("document",)
    inlines = [ChildChunkInline]


@admin.register(ChildChunk)
class ChildChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "parent", "created_at")
    list_filter = ("document",)
    search_fields = ("id", "document__filename", "parent__id")
    readonly_fields = ("id", "created_at")
    raw_id_fields = ("document", "parent")
