"""Tests for retrieval strategies.

Basic tests to verify retrieval functionality. Note: These tests require
Qdrant to be running for dense/hybrid retrieval.
"""

from __future__ import annotations

import pytest
from llama_index.core import Document

from rag_eval.retrieval.sparse import SparseRetriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    docs = [
        Document(
            text="Python is a high-level programming language.",
            doc_id="doc1",
            metadata={"topic": "programming"},
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence.",
            doc_id="doc2",
            metadata={"topic": "ai"},
        ),
        Document(
            text="Deep learning uses neural networks with multiple layers.",
            doc_id="doc3",
            metadata={"topic": "ai"},
        ),
        Document(
            text="JavaScript is used for web development.",
            doc_id="doc4",
            metadata={"topic": "programming"},
        ),
        Document(
            text="Natural language processing helps computers understand human language.",
            doc_id="doc5",
            metadata={"topic": "ai"},
        ),
        Document(
            text="React is a JavaScript library for building user interfaces.",
            doc_id="doc6",
            metadata={"topic": "programming"},
        ),
        Document(
            text="Transformers are a type of neural network architecture.",
            doc_id="doc7",
            metadata={"topic": "ai"},
        ),
        Document(
            text="Python is popular for data science and machine learning.",
            doc_id="doc8",
            metadata={"topic": "programming"},
        ),
        Document(
            text="Docker is a platform for containerizing applications.",
            doc_id="doc9",
            metadata={"topic": "devops"},
        ),
        Document(
            text="Kubernetes orchestrates containerized applications.",
            doc_id="doc10",
            metadata={"topic": "devops"},
        ),
    ]
    return docs


class TestSparseRetriever:
    """Tests for BM25 sparse retrieval."""

    def test_index_and_retrieve(self, sample_documents: list[Document]) -> None:
        """Test basic indexing and retrieval."""
        retriever = SparseRetriever()
        retriever.index(sample_documents)

        results = retriever.retrieve("Python programming", top_k=3)

        # Should return results
        assert len(results) > 0
        assert len(results) <= 3

        # Results should have required fields
        for result in results:
            assert result.doc_id
            assert result.text
            assert result.score >= 0

    def test_retrieve_relevant_document(self, sample_documents: list[Document]) -> None:
        """Test that relevant documents are retrieved."""
        retriever = SparseRetriever()
        retriever.index(sample_documents)

        # Query for Python - should find doc1 or doc8
        results = retriever.retrieve("Python", top_k=5)
        doc_ids = [r.doc_id for r in results]

        assert "doc1" in doc_ids or "doc8" in doc_ids

    def test_retrieve_before_index_raises(self) -> None:
        """Test that retrieving before indexing raises an error."""
        retriever = SparseRetriever()

        with pytest.raises(ValueError, match="Index not created"):
            retriever.retrieve("test query")

    def test_top_k_respected(self, sample_documents: list[Document]) -> None:
        """Test that top_k parameter is respected."""
        retriever = SparseRetriever()
        retriever.index(sample_documents)

        results = retriever.retrieve("programming", top_k=2)
        assert len(results) == 2

    def test_results_sorted_by_score(self, sample_documents: list[Document]) -> None:
        """Test that results are sorted by score descending."""
        retriever = SparseRetriever()
        retriever.index(sample_documents)

        results = retriever.retrieve("machine learning", top_k=5)

        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        retriever = SparseRetriever()
        assert retriever.strategy_name == "sparse"


class TestDenseRetriever:
    """Tests for dense vector retrieval.

    Note: These tests require Qdrant to be running.
    """

    @pytest.mark.skipif(
        True,  # Skip by default - requires Qdrant
        reason="Requires Qdrant running - run manually with --run-integration",
    )
    def test_index_and_retrieve(self, sample_documents: list[Document]) -> None:
        """Test basic indexing and retrieval with Qdrant."""
        retriever = DenseRetriever(
            collection_name="test_dense",
            embedding_model="all-MiniLM-L6-v2",
            qdrant_host="localhost",
            qdrant_port=6333,
        )

        try:
            retriever.index(sample_documents)
            results = retriever.retrieve("Python programming", top_k=3)

            assert len(results) > 0
            assert len(results) <= 3

            for result in results:
                assert result.doc_id
                assert result.text
                assert result.score >= 0

        finally:
            # Clean up
            retriever.clear_collection()

    @pytest.mark.skipif(
        True,
        reason="Requires Qdrant running - run manually with --run-integration",
    )
    def test_semantic_similarity(self, sample_documents: list[Document]) -> None:
        """Test that semantically similar documents are retrieved."""
        retriever = DenseRetriever(
            collection_name="test_semantic",
            embedding_model="all-MiniLM-L6-v2",
        )

        try:
            retriever.index(sample_documents)

            # Query for AI - should find AI-related docs (doc2, doc3, doc5, doc7)
            results = retriever.retrieve("artificial intelligence neural networks", top_k=3)
            doc_ids = [r.doc_id for r in results]

            # At least one AI doc should be in top 3
            ai_docs = {"doc2", "doc3", "doc5", "doc7"}
            assert any(doc_id in ai_docs for doc_id in doc_ids)

        finally:
            retriever.clear_collection()

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        retriever = DenseRetriever()
        assert retriever.strategy_name == "dense"


class TestHybridRetriever:
    """Tests for hybrid RRF retrieval."""

    @pytest.mark.skipif(
        True,
        reason="Requires Qdrant running - run manually with --run-integration",
    )
    def test_hybrid_retrieval(self, sample_documents: list[Document]) -> None:
        """Test hybrid retrieval with RRF fusion."""
        dense = DenseRetriever(
            collection_name="test_hybrid_dense",
            embedding_model="all-MiniLM-L6-v2",
        )
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse, rrf_k=60)

        try:
            hybrid.index(sample_documents)
            results = hybrid.retrieve("machine learning Python", top_k=5)

            assert len(results) > 0
            assert len(results) <= 5

            # Results should have RRF scores
            for result in results:
                assert result.doc_id
                assert result.text
                assert result.score > 0

        finally:
            dense.clear_collection()

    @pytest.mark.skipif(
        True,
        reason="Requires Qdrant running - run manually with --run-integration",
    )
    def test_rrf_fusion_combines_results(self, sample_documents: list[Document]) -> None:
        """Test that RRF actually combines results from both retrievers."""
        dense = DenseRetriever(
            collection_name="test_rrf",
            embedding_model="all-MiniLM-L6-v2",
        )
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse)

        try:
            hybrid.index(sample_documents)

            # Get results from each retriever individually
            dense_results = dense.retrieve("Python", top_k=3)
            sparse_results = sparse.retrieve("Python", top_k=3)

            # Get hybrid results
            hybrid_results = hybrid.retrieve("Python", top_k=5)

            # Hybrid should potentially include docs from both retrievers
            hybrid_doc_ids = {r.doc_id for r in hybrid_results}
            dense_doc_ids = {r.doc_id for r in dense_results}
            sparse_doc_ids = {r.doc_id for r in sparse_results}

            # Hybrid results should include some docs from the component retrievers
            assert len(hybrid_doc_ids & (dense_doc_ids | sparse_doc_ids)) > 0

        finally:
            dense.clear_collection()

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        dense = DenseRetriever()
        sparse = SparseRetriever()
        hybrid = HybridRetriever(dense, sparse)

        assert hybrid.strategy_name == "hybrid"
