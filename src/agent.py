from re import search
from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
        search_result = self.store.search(query=question, top_k=top_k)
        if not search_result:
            return "Tôi xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở kiến thức để trả lời câu hỏi này."

        context_text = "\n---\n".join([res["content"] for res in search_result])

        prompt = f"""Bạn là một trợ lý thông minh. Hãy sử dụng PHẦN NGỮ CẢNH được cung cấp dưới đây để trả lời CÂU HỎI của người dùng. 
        Nếu thông tin trong ngữ cảnh không đủ để trả lời, hãy nói rằng bạn không biết, đừng cố tự tạo ra câu trả lời.

        PHẦN NGỮ CẢNH:
        {context_text}

        CÂU HỎI: 
        {question}

        CÂU TRẢ LỜI:"""

        answer = self.llm_fn(prompt)
        
        return answer