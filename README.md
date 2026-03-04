# RAG Evaluation

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-CC785C?style=flat&logo=anthropic&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-gemma3:4b-black?style=flat&logo=ollama&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Store-DC244C?style=flat&logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Management-60A5FA?style=flat&logo=poetry&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-Generation%20Eval-FF6B6B?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)


A rigorous evaluation framework for comparing RAG (Retrieval-Augmented Generation) retrieval strategies and LLM providers, built with production engineering practices.

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
| Fixed + Dense | fixed | dense | 0.6059 | 0.6434 | 0.7933 | 300 |
| Recursive + Dense | recursive | dense | 0.6285 | 0.6532 | 0.7633 | 300 |
| Hybrid (BM25 + Dense) | recursive | hybrid_rrf | 0.5454 | 0.5835 | 0.7433 | 300 |

### LLM Provider Comparison

*Using best retrieval strategy (determined from above)*

```bash
python scripts/run_benchmark.py --mode llm_comparison --output results/llm_comparison.json
python scripts/generate_report.py results/llm_comparison.json
```

| Provider | Model | Faithfulness | Answer Relevance | Samples |
|---|---|---|---|---|
| OpenAI | gpt-5-nano | 0.6466 | 0.5186 | 50 |
| Anthropic | claude-haiku-4-5 | 0.8523 | 0.2804* | 41 |
| Ollama | gemma3:4b (local) | 0.7282 | 0.5654 | 50 |

\* See Spot Check Analysis for explanation of low answer relevance score.

## Key Findings

1. **Hybrid retrieval does not always win.** Despite combining BM25 and dense retrieval via RRF fusion, hybrid underperformed dense-only strategies on this dataset. SciFact queries are short and precise — hybrid retrieval adds value primarily on longer, ambiguous queries where sparse and dense signals are complementary.

2. **Faithfulness and answer relevance measure different failure modes.** High faithfulness means the model stays grounded in retrieved context. High answer relevance means the response addresses the question. A model can score high on one and low on the other — Claude-Haiku demonstrates this clearly.

3. **RAGAS penalises calibrated responses.** Claude-Haiku achieved the highest faithfulness (0.85) but lowest answer relevance (0.28). Manual inspection showed this is a scoring artifact: Claude's structured "insufficient evidence" responses generate no evaluable statements, scoring 0.0 on relevance regardless of actual quality. This is a known limitation of LLM-as-judge metrics.

4. **High answer relevance does not equal correctness.** Gemma3:4b scored highest on answer relevance but hallucinated in at least one query — constructing a confident "Yes" answer to a question where the correct response was "insufficient evidence," by stitching together loosely related evidence. Automated metrics rewarded the confident wrong answer over the correct cautious one.

5. **Retrieval failures propagate directly to generation.** Query 1 returned completely irrelevant chunks (rickets, carbon nanotubes) for a question about 0-dimensional biomaterials. All three LLMs correctly identified insufficient context — but all scored 0.0 on answer relevance as a result. Generation quality is bounded by retrieval quality regardless of LLM capability.

## Spot Check Analysis

Automated metrics alone are insufficient to evaluate RAG quality. We manually inspected outputs across 5 queries to validate benchmark findings.

| Query | OpenAI | Anthropic | Gemma |
|-------|--------|-----------|-------|
| 0-dim biomaterials | Correct (no info) | Correct (no info) | Correct (no info) |
| ALDH1 breast cancer | Correct | Correct | Partial (quotes chunk verbatim) |
| ACE inhibitors | Partially correct | Correct | **Hallucination** |
| Visual impairment screening | Correct | Correct | Correct (quotes verbatim) |
| Bone marrow macrophages | Correct | Correct | Partially correct (factual error) |

### Illustrative Example: Gemma Hallucination (Query 3)

**Query:** *Angiotensin converting enzyme inhibitors are associated with increased risk for functional renal insufficiency.*

The correct answer based on retrieved context is "insufficient evidence." OpenAI and Claude both respond accordingly. Gemma constructs a confident "Yes" by stitching together loosely related evidence across documents — ARB hyperkalemia risk, contrast agent nephrotoxicity, and diabetic interactions — none of which directly support the claim. It scores 0.93 on answer relevance despite being wrong.

This is a concrete example of why answer relevance is an unreliable standalone quality signal. Run `scripts/inspect_outputs.py` to reproduce the full output.


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
rag-eval/
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
│   ├── test_retrieval.py          # Retrieval strategy tests
│   └── test_llm_providers.py      # LLM provider tests (mocked)
├── scripts/
│   ├── run_benchmark.py           # Main benchmark runner, saves results to JSON
│   ├── generate_report.py         # Reads JSON results, prints formatted table
│   └── inspect_outputs.py         # Manual spot-check of LLM outputs per query
├── docs/
│   ├── prd.md                     # Product requirements document
│   └── DIAGRAMS.md                # Architecture and flow diagrams
├── results/                       # Benchmark output JSONs (git-ignored)
├── docker-compose.yml             # Qdrant + app services
├── Dockerfile                     # App container
├── pyproject.toml                 # Poetry dependencies
├── poetry.lock                    # Locked dependency versions
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

# LLM Model Configuration (per provider)
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-haiku-4-5
OLLAMA_MODEL=llama3.2

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
  author = {Anshumaan Ravi},
  year = {2026},
  url = {https://github.com/anshu-ravi/rag-eval}
}
```
