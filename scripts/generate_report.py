"""Generate formatted markdown report from benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def format_retrieval_results(results: list[dict[str, Any]]) -> str:
    """Format retrieval benchmark results as markdown table.

    Args:
        results: List of retrieval benchmark results.

    Returns:
        Formatted markdown table.
    """
    table = """
| Strategy | Chunker | Retriever | MRR@10 | NDCG@10 | Hit Rate@10 | Queries |
|---|---|---|---|---|---|---|
"""

    for result in results:
        row = (
            f"| {result['strategy']} | {result['chunker']} | {result['retriever']} | "
            f"{result['mrr_at_10']:.4f} | {result['ndcg_at_10']:.4f} | "
            f"{result['hit_rate_at_10']:.4f} | {result['n_queries']} |\n"
        )
        table += row

    return table


def format_llm_results(results: list[dict[str, Any]]) -> str:
    """Format LLM comparison results as markdown table.

    Args:
        results: List of LLM comparison results.

    Returns:
        Formatted markdown table.
    """
    table = """
| Provider | Model | Faithfulness | Answer Relevance | Samples |
|---|---|---|---|---|
"""

    for result in results:
        row = (
            f"| {result['provider']} | {result['model']} | "
            f"{result['faithfulness']:.4f} | {result['answer_relevance']:.4f} | "
            f"{result['n_samples']} |\n"
        )
        table += row

    return table


def generate_report(input_path: str) -> str:
    """Generate markdown report from benchmark results JSON.

    Args:
        input_path: Path to results JSON file.

    Returns:
        Formatted markdown report.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    experiment_type = data.get("experiment", "unknown")
    timestamp = data.get("timestamp", "unknown")
    results = data.get("results", [])

    report = f"""# Benchmark Results

**Experiment**: {experiment_type}
**Dataset**: {data.get('dataset', 'unknown')}
**Timestamp**: {timestamp}

"""

    if experiment_type == "retrieval_benchmark":
        report += "## Retrieval Strategy Comparison\n"
        report += format_retrieval_results(results)

        # Add configuration
        config = data.get("config", {})
        report += f"""
### Configuration
- Embedding model: {config.get('embedding_model', 'N/A')}
- Top-k: {config.get('top_k', 'N/A')}
- Chunk size: {config.get('chunk_size', 'N/A')}
- Chunk overlap: {config.get('chunk_overlap', 'N/A')}
"""

    elif experiment_type == "llm_comparison":
        report += f"## LLM Provider Comparison\n"
        report += f"**Retrieval Strategy**: {data.get('retrieval_strategy', 'N/A')}\n\n"
        report += format_llm_results(results)

        # Add configuration
        config = data.get("config", {})
        report += f"""
### Configuration
- Top-k: {config.get('top_k', 'N/A')}
- Sampled queries: {config.get('n_sampled_queries', 'N/A')}
- Random seed: {config.get('random_seed', 'N/A')}
"""

    return report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "input",
        type=str,
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, prints to stdout if not provided)",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_report(args.input)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
