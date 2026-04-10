from __future__ import annotations

from pathlib import Path
from src.chunking import ChunkingStrategyComparator

DOCUMENTS_TO_ANALYZE = [
    "book/caykhe.txt",
    "book/hoguom.txt",
    "book/nguulangchucnu.txt",
    "book/sodua.txt",
    "book/thachsanh.txt"
]

def run_analysis():
    """
    Runs the chunking strategy comparison and prints the results
    in a Markdown table format for your report.
    """
    comparator = ChunkingStrategyComparator()
    
    print("## Baseline Analysis Results")
    print()
    print("| Tài liệu | Strategy | Chunk Count | Avg Length |")
    print("|---|---|---|---|")

    for doc_path_str in DOCUMENTS_TO_ANALYZE:
        doc_path = Path(doc_path_str)
        if not doc_path.exists():
            print(f"| {doc_path.name} | *File not found* | | |")
            continue

        text = doc_path.read_text(encoding="utf-8")
        results = comparator.compare(text)

        for i, (strategy_name, stats) in enumerate(results.items()):
            doc_name = doc_path.name if i == 0 else ""
            avg_len = f"{stats.get('avg_length', 0):.1f}"
            print(f"| {doc_name} | {strategy_name} | {stats.get('count', 0)} | {avg_len} |")

if __name__ == "__main__":
    run_analysis()
