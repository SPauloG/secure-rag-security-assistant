"""
Covers the parts of the upsert step that run without network access: how chunks are
grouped into API requests and what metadata travels with each vector.
"""
from src.ingestion.ingest import (
    EMBED_BATCH_MAX_CHARS,
    EMBED_BATCH_MAX_TEXTS,
    _embedding_batches,
    _vector_metadata,
)


def chunk(text: str = "x", **overrides) -> dict:
    base = {
        "id": "src#p1#0",
        "text": text,
        "source": "doc.pdf",
        "title": "A Document",
        "page": 1,
        "section": None,
    }
    return base | overrides


def test_metadata_carries_everything_a_citation_needs():
    metadata = _vector_metadata(chunk(text="the passage"))
    assert metadata["text"] == "the passage"
    assert metadata["source"] == "doc.pdf"
    assert metadata["title"] == "A Document"
    assert metadata["page"] == 1


def test_metadata_omits_empty_fields_because_pinecone_rejects_nulls():
    pdf = _vector_metadata(chunk(page=3, section=None))
    html = _vector_metadata(chunk(page=None, section="How to Prevent"))

    assert "section" not in pdf
    assert "page" not in html
    assert html["section"] == "How to Prevent"
    assert None not in pdf.values() and None not in html.values()


def test_batches_stay_within_the_request_size_limit():
    chunks = [chunk(text="y" * 5000) for _ in range(200)]
    batches = _embedding_batches(chunks)

    assert sum(len(b) for b in batches) == len(chunks)
    for batch in batches:
        assert len(batch) <= EMBED_BATCH_MAX_TEXTS
        assert sum(len(c["text"]) for c in batch) <= EMBED_BATCH_MAX_CHARS


def test_batches_split_on_character_budget_not_only_on_count():
    huge = [chunk(text="z" * (EMBED_BATCH_MAX_CHARS // 4)) for _ in range(8)]
    batches = _embedding_batches(huge)

    assert len(batches) > 1
    assert all(len(b) < EMBED_BATCH_MAX_TEXTS for b in batches)


def test_a_single_oversized_chunk_still_gets_its_own_batch():
    chunks = [chunk(text="a"), chunk(text="b" * (EMBED_BATCH_MAX_CHARS + 1)), chunk(text="c")]
    batches = _embedding_batches(chunks)

    assert sum(len(b) for b in batches) == 3
    assert any(len(b) == 1 for b in batches)


def test_no_batches_for_no_chunks():
    assert _embedding_batches([]) == []
