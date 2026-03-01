"""Abstract base class for retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core import Document


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    doc_id: str
    score: float
    text: str


class BaseRetriever(ABC):
    """Base class for all retrieval strategies."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of documents to index.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieval results sorted by relevance score (descending).
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of the retrieval strategy."""
        pass
