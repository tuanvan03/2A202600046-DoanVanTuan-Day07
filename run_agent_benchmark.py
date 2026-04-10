from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker, SentenceChunker, FixedSizeChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
load_dotenv(override=False)

BENCHMARK_QUERIES = {
    "Sọ Dừa có ngoại hình như thế nào từ khi sinh ra?": "Là một khối thịt đỏ hỏn, không tay không chân, tròn lăn lóc giống như một quả dừa.",
    "Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại?": "Lý Thông vốn là kẻ tiểu nhân tráo trở, thấy Thạch Sanh thật thà khoẻ mạnh nên lợi dụng để cướp công giết chằn tinh nhằm tiến thân.",
    "Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào?": "Gắn liền trực tiếp với cuộc chiến của vua Lê Lợi (mệnh danh Bình Định Vương) mượn Gươm của Thần Kim Quy đánh tan quân Minh.",
    "Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ đâu?": "Bắt nguồn từ sự cấm cản của Ngọc Hoàng vì ranh giới Tiên - Phàm và trách nhiệm chốn tiên giới bị bỏ bê.",
    "Bài học rõ nét nhất từ câu chuyện Cây Khế?": "Lòng tham vô đáy (như người anh) sẽ chuốc lấy sự hủy diệt, còn sự chia sẻ yêu thương sẽ đơm bông kết trái bền vững.",
}

STORY_METADATA = {
    "sodua": {
        "story_title": ["Sọ Dừa"],
        "story_type": "cổ tích",
        "origin": "Việt Nam",
        "themes": ["phép thuật", "tình yêu", "lòng tốt"],
        "main_characters": ["Sọ Dừa", "Phú Ông", "cô út"],
    },
    "thachsanh": {
        "story_title": ["Thạch Sanh"],
        "story_type": "cổ tích",
        "origin": "Việt Nam",
        "themes": ["anh hùng", "phép thuật", "thiện ác"],
        "main_characters": ["Thạch Sanh", "Lý Thông", "công chúa"],
    },
    "hoguom": {
        "story_title": ["Sự tích Hồ Gươm", "Hồ Hoàn Kiếm"],
        "story_type": "truyền thuyết",
        "origin": "Việt Nam",
        "themes": ["lịch sử", "yêu nước", "thần linh"],
        "main_characters": ["Lê Lợi", "Rùa Vàng", "Lê Thận"],
    },
    "nguulangchucnu": {
        "story_title": ["Ngưu Lang Chức Nữ"],
        "story_type": "truyền thuyết",
        "origin": "Trung Quốc",
        "themes": ["tình yêu", "chia ly", "thiên đình"],
        "main_characters": ["Ngưu Lang", "Chức Nữ", "Ngọc Hoàng"],
    },
    "caykhe": {
        "story_title": ["Cây Khế"],
        "story_type": "cổ tích",
        "origin": "Việt Nam",
        "themes": ["tham lam", "thiện ác", "lòng tốt"],
        "main_characters": ["người anh", "người em", "chim phượng hoàng"],
    },
}

def load_documents_from_dir(dir_path: str) -> list[Document]:
    """Loads all .txt files from a directory into Document objects."""
    docs = []
    for file_path in Path(dir_path).glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        
        # Combine base metadata with detailed story metadata
        file_stem = file_path.stem
        story_meta = STORY_METADATA.get(file_stem, {})
        combined_meta = {"source": str(file_path), **story_meta}

        doc = Document(
            id=file_stem,
            content=content,
            metadata=combined_meta,
        )
        docs.append(doc)
    return docs

def openai_llm(prompt: str) -> str:
    """Calls the OpenAI API to get a response."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Failed to call OpenAI API: {e}"

def run_benchmark():
    """
    Runs the benchmark queries against the KnowledgeBaseAgent and
    prints the results in a Markdown table format.
    """
    print("--- Agent Benchmark ---")

    # 1. Load Documents
    docs = load_documents_from_dir("book")
    if not docs:
        print("No documents found in 'book/' directory. Please add your .txt files.")
        return

    print(f"Loaded {len(docs)} documents from 'book/' directory.")

    # 2. Chunk Documents
    # You can swap this with your chosen chunking strategy
    # chunker = RecursiveChunker(chunk_size=500)
    chunker = SentenceChunker(max_sentences_per_chunk=4)
    # chunker = FixedSizeChunker(chunk_size=800, overlap=100)
    chunked_docs = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"{doc.id}-chunk{i}"
            chunked_docs.append(
                Document(
                    id=chunk_id,
                    content=chunk_content,
                    metadata={"doc_id": doc.id, "source": doc.metadata["source"]},
                )
            )
    
    print(f"Created {len(chunked_docs)} chunks.")
    # print(chunked_docs[:3])

    # 3. Setup Embedder
    embedder = _mock_embed
    print(f"Using embedding backend: {getattr(embedder, '_backend_name', 'mock')}")

    # 4. Setup Store and Agent
    llm_function = openai_llm
    print("Using OpenAI LLM for generation.")

    store = EmbeddingStore(embedding_fn=embedder)
    store.add_documents(chunked_docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_function)

    print("\n--- Benchmark Results ---")
    print("| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |")
    print("|---|---|---|---|---|---|")

    if not BENCHMARK_QUERIES:
        print("| | *No benchmark queries defined. Please add them to `BENCHMARK_QUERIES`.* | | | | |")
        return

    # 5. Run Queries
    for i, query in enumerate(BENCHMARK_QUERIES.keys(), 1):
        # Get top retrieved chunk for analysis
        search_results = store.search(query, top_k=1)
        top_chunk_summary = "N/A"
        score = "N/A"
        if search_results:
            top_result = search_results[0]
            content_preview = top_result['content'][:60].replace("\n", " ").replace("|", "\\|")
            top_chunk_summary = f"[{top_result['metadata']['source']}] {content_preview}..."
            score = f"{top_result.get('score', 0):.3f}"

        # Get agent's answer
        answer = agent.answer(query, top_k=5)
        answer_summary = answer.replace("\n", " ").replace("|", "\\|")

        print(f"| {i} | {query} | {top_chunk_summary} | {score} | | {answer_summary} |")

if __name__ == "__main__":
    run_benchmark()
