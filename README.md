# production-rag-eval

A production-grade RAG (Retrieval-Augmented Generation) system with rigorous evaluation framework, model-agnostic LLM abstraction, and comprehensive benchmark results comparing retrieval strategies and LLM providers.

## Problem Statement

RAG systems are widely used but rarely evaluated rigorously. Most implementations:
- Lack systematic comparison of retrieval strategies
- Don't measure generation quality with established metrics
- Are tightly coupled to specific LLM providers
- Have no reproducible benchmarks

This project addresses these gaps by providing:
- **Rigorous evaluation**: Standard IR metrics (MRR, NDCG, Hit Rate) + generation quality metrics (Faithfulness, Answer Relevance)
- **Retrieval strategy comparison**: Fixed chunking, recursive chunking, and hybrid BM25+dense retrieval
- **Model-agnostic design**: Unified interface for OpenAI, Anthropic, and Ollama
- **Reproducible benchmarks**: Standardized dataset (BEIR SciFact) with fixed random seeds

## Approach

### Retrieval Strategies

1. **Fixed Chunking + Dense**: Token-based chunking (512 tokens) with sentence-transformers embeddings
2. **Recursive Chunking + Dense**: Semantic-aware splitting on paragraph/sentence boundaries
3. **Hybrid (BM25 + Dense)**: Reciprocal Rank Fusion (RRF) of sparse and dense retrieval

### Evaluation Methodology

**Retrieval Metrics** (computed on full test set - 300 queries):
- **MRR@10**: Mean Reciprocal Rank - position of first relevant document
- **NDCG@10**: Normalized Discounted Cumulative Gain - ranking quality with graded relevance
- **Hit Rate@10**: Percentage of queries with at least one relevant document in top-10

**Generation Metrics** (computed on 50 sampled queries via RAGAS):
- **Faithfulness**: Are generated answers supported by retrieved context? (LLM-as-judge)
- **Answer Relevance**: Do answers address the question asked? (LLM-as-judge)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (for dependency management)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd production-rag-eval

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running with Docker

```bash
# Start Qdrant vector store
docker-compose up -d qdrant

# Run retrieval benchmark
docker-compose run --rm app --mode retrieval --output results/retrieval.json

# Run LLM comparison
docker-compose run --rm app --mode llm_comparison --output results/llm_comparison.json
```

### Running Locally

```bash
# Start Qdrant
docker-compose up -d qdrant

# Run retrieval benchmark
python scripts/run_benchmark.py --mode retrieval --output results/retrieval.json

# Run LLM comparison
python scripts/run_benchmark.py --mode llm_comparison --output results/llm_comparison.json

# Generate markdown report
python scripts/generate_report.py results/retrieval.json
```

## Benchmark Results

### Retrieval Strategy Comparison

*Run the benchmark to populate these results:*

```bash
python scripts/run_benchmark.py --mode retrieval --output results/retrieval.json
python scripts/generate_report.py results/retrieval.json
```

| Strategy | Chunker | Retriever | MRR@10 | NDCG@10 | Hit Rate@10 | Queries |
|---|---|---|---|---|---|---|
| Fixed + Dense | fixed | dense | TBD | TBD | TBD | 300 |
| Recursive + Dense | recursive | dense | TBD | TBD | TBD | 300 |
| Hybrid (BM25 + Dense) | recursive | hybrid_rrf | TBD | TBD | TBD | 300 |

### LLM Provider Comparison

*Using best retrieval strategy (determined from above)*

```bash
python scripts/run_benchmark.py --mode llm_comparison --output results/llm_comparison.json
python scripts/generate_report.py results/llm_comparison.json
```

| Provider | Model | Faithfulness | Answer Relevance | Samples |
|---|---|---|---|---|
| OpenAI | gpt-4o-mini | TBD | TBD | 50 |
| Anthropic | claude-haiku-4-5 | TBD | TBD | 50 |
| Ollama | llama3.2 (local) | TBD | TBD | 50 |

## Key Findings

*Results to be filled in after running benchmarks:*

1. **Retrieval Strategy Impact**: [Comparison of strategies on NDCG@10]
2. **Hybrid vs. Dense-Only**: [Performance gain from RRF fusion]
3. **LLM Provider Comparison**: [Faithfulness and relevance across providers]
4. **Cost-Performance Tradeoff**: [Cost per query vs. quality metrics]
5. **Local vs. Cloud LLMs**: [Ollama performance vs. commercial APIs]

## Architecture

### Model-Agnostic Design

The core abstraction is `BaseLLMProvider`:

```python
class BaseLLMProvider(ABC):
    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion for the given prompt."""
        pass
```

All LLM providers (OpenAI, Anthropic, Ollama) implement this interface. Provider selection is config-driven via `LLM_PROVIDER` environment variable - no code changes needed to swap providers.

### Retrieval Abstraction

Similarly, `BaseRetriever` provides a unified interface for all retrieval strategies:

```python
class BaseRetriever(ABC):
    def index(self, documents: list[Document]) -> None:
        """Index documents for retrieval."""
        pass

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant documents."""
        pass
```

This enables fair comparison across strategies using identical evaluation logic.

### RAG Pipeline

The `RAGPipeline` orchestrates retrieval and generation:

1. Retrieve top-k documents for the query
2. Format documents as context
3. Create prompt combining context + question
4. Generate answer using configured LLM provider
5. Return answer with provenance (retrieved documents + metadata)

## Repository Structure

```
production-rag-eval/
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings, loaded from .env
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # BEIR SciFact loader and preprocessor
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base class for chunkers
│       │   ├── fixed.py           # Fixed-size token chunking
│       │   ├── recursive.py       # Recursive character text splitting
│       │   └── semantic.py        # Semantic chunking via embedding similarity
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base class for retrievers
│       │   ├── dense.py           # Dense retrieval (Qdrant + sentence-transformers)
│       │   ├── sparse.py          # BM25 (rank-bm25)
│       │   └── hybrid.py          # Hybrid: BM25 + dense with RRF fusion
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract LLM provider interface
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   └── ollama_provider.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── rag_pipeline.py    # Orchestrates retrieval + generation
│       └── evaluation/
│           ├── __init__.py
│           ├── retrieval_metrics.py  # MRR, NDCG@10, Hit Rate (from scratch)
│           └── generation_metrics.py # Faithfulness, Answer Relevance (RAGAS)
├── tests/
│   ├── test_chunking.py           # Chunking strategy tests
│   ├── test_metrics.py            # Metric implementation tests
│   └── test_llm_providers.py      # LLM provider tests (mocked)
├── scripts/
│   ├── run_benchmark.py           # Main benchmark runner, saves results to JSON
│   └── generate_report.py         # Reads JSON results, prints formatted table
├── docker-compose.yml             # Qdrant + app services
├── Dockerfile                     # App container
├── pyproject.toml                 # Poetry dependencies
├── .env.example                   # Example environment configuration
└── README.md
```

## Configuration

All configuration is managed via `.env` file (using pydantic-settings):

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434

# LLM Provider
LLM_PROVIDER=openai           # openai | anthropic | ollama
LLM_MODEL=gpt-4o-mini         # optional override

# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embeddings and Retrieval
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=10
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=rag_eval --cov-report=html
```

## Development

```bash
# Install with dev dependencies
poetry install

# Format code
black src/ tests/

# Lint
ruff src/ tests/

# Type checking
mypy src/
```

## Dataset

**BEIR SciFact**: Scientific fact verification dataset
- 5,183 scientific paper abstracts as corpus
- 300 test queries with ground-truth relevance judgments
- Automatically downloaded on first run to `./datasets/`

## Implementation Notes

### Metrics Implementation

Retrieval metrics (MRR, NDCG, Hit Rate) are **implemented from scratch** (not delegated to BEIR's evaluator) to demonstrate understanding of the evaluation methodology. See `src/rag_eval/evaluation/retrieval_metrics.py`.

### RRF Fusion

Reciprocal Rank Fusion is **implemented manually** using the formula:

```
score(d) = Σ 1 / (k + rank_i(d))
```

where `k=60` and `rank_i(d)` is the rank of document `d` in retrieval method `i`. See `src/rag_eval/retrieval/hybrid.py`.

### Reproducibility

- Random seed fixed to `42` for query sampling in generation evaluation
- Same dataset split (BEIR test set) used consistently
- All configuration parameters logged in results JSON

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Citation

If you use this work, please cite:

```bibtex
@software{production_rag_eval,
  title = {production-rag-eval: Rigorous RAG System Evaluation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/production-rag-eval}
}
```
