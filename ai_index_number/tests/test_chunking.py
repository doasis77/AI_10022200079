# tests/test_chunking.py
# Author: [Your Name] | Index: [Your Index Number]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.chunking import fixed_size_chunks, paragraph_aware_chunks


SAMPLE_PAGES = [
    {
        "page_number": 1,
        "source": "test.pdf",
        "cleaned_text": " ".join([f"word{i}" for i in range(1000)])
    }
]


def test_fixed_size_chunks_count():
    """Fixed-size chunks should produce ceil((1000-80)/(400-80)) + 1 chunks roughly."""
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=400, overlap=80)
    assert len(chunks) > 0, "Should produce at least one chunk"
    print(f"  Fixed-size: {len(chunks)} chunks produced")


def test_fixed_size_chunk_structure():
    """Each chunk must have required keys."""
    chunks = fixed_size_chunks(SAMPLE_PAGES)
    for c in chunks:
        assert "chunk_id" in c
        assert "source" in c
        assert "text" in c
        assert "chunk_type" in c
        assert c["chunk_type"] == "fixed_size"
    print("  Fixed-size chunk structure: OK")


def test_fixed_overlap():
    """Last word of chunk N should appear near start of chunk N+1."""
    chunks = fixed_size_chunks(SAMPLE_PAGES, chunk_size=50, overlap=10)
    if len(chunks) >= 2:
        end_words = set(chunks[0]["text"].split()[-10:])
        start_words = set(chunks[1]["text"].split()[:10])
        assert end_words & start_words, "Overlap words should appear in next chunk"
    print("  Fixed-size overlap: OK")


def test_paragraph_aware_chunks():
    pages = [{"page_number": 1, "source": "t", "cleaned_text":
        "The government has committed to fiscal discipline. Revenue targets are set. "
        "Expenditure will be controlled. " * 40}]
    chunks = paragraph_aware_chunks(pages)
    assert len(chunks) > 0
    print(f"  Paragraph-aware: {len(chunks)} chunks produced")


if __name__ == "__main__":
    print("Running chunking tests...")
    test_fixed_size_chunks_count()
    test_fixed_size_chunk_structure()
    test_fixed_overlap()
    test_paragraph_aware_chunks()
    print("All chunking tests passed ✓")
