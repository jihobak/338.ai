import datetime
from typing import List, Optional, Tuple

import weave
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel

from bot338.rag.query_handler import QueryEnhancer
from bot338.rag.response_synthesis import ResponseSynthesizer, StreamResponseSynthesizer
from bot338.rag.retrieval import FusionRetrieval
from bot338.retriever import VectorStore
from bot338.utils import Timer, get_logger


logger = get_logger(__name__)


class RAGPipelineOutput(BaseModel):
    question: str
    answer: str
    sources: str = ""
    source_documents: str
    system_prompt: str
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    time_taken: float
    start_time: datetime.datetime
    end_time: datetime.datetime
    api_call_statuses: dict = {}


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 15,
        search_type: str = "mmr",
        multilingual_reranker_model: str = "rerank-multilingual-v3.0",
        response_synthesizer_model: str = "gpt-4o-2024-08-06",
        response_synthesizer_temperature: float = 0.1,
        response_synthesizer_fallback_model: str = "gpt-4o-2024-08-06",
        response_synthesizer_fallback_temperature: float = 0.1,
    ):
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer()
        self.retrieval = FusionRetrieval(
            vector_store=vector_store,
            top_k=top_k,
            search_type=search_type,
            multilingual_reranker_model=multilingual_reranker_model,
        )
        self.response_synthesizer = ResponseSynthesizer(
            model=response_synthesizer_model,
            temperature=response_synthesizer_temperature,
            fallback_model=response_synthesizer_fallback_model,
            fallback_temperature=response_synthesizer_fallback_temperature,
        )

    @weave.op()
    def __call__(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] | None = None,
        reranking: bool = False,
    ) -> RAGPipelineOutput:
        if chat_history is None:
            chat_history = []

        with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
            enhanced_query = self.query_enhancer(
                {"query": question, "chat_history": chat_history}
            )

        with Timer() as retrieval_tb:
            retrieval_results = self.retrieval(enhanced_query, reranking=reranking)

        with get_openai_callback() as response_cb, Timer() as response_tb:
            response = self.response_synthesizer(retrieval_results)

        output = RAGPipelineOutput(
            question=enhanced_query["standalone_query"],
            answer=response["response"],
            # sources="\n".join(
            #     [item.metadata["source"] for item in retrieval_results["context"]]
            # ),
            source_documents=response["context_str"],
            system_prompt=response["response_prompt"],
            model=response["response_model"],
            total_tokens=query_enhancer_cb.total_tokens + response_cb.total_tokens,
            prompt_tokens=query_enhancer_cb.prompt_tokens + response_cb.prompt_tokens,
            completion_tokens=query_enhancer_cb.completion_tokens
            + response_cb.completion_tokens,
            time_taken=query_enhancer_tb.elapsed
            + retrieval_tb.elapsed
            + response_tb.elapsed,
            start_time=query_enhancer_tb.start,
            end_time=response_tb.stop,
            api_call_statuses={
                # "web_search_success": retrieval_results["web_search_success"],
            },
        )

        return output


class StreamRAGPipeline(RAGPipeline):
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 15,
        search_type: str = "mmr",
        fetch_k: int = 10,
        lambda_mult: Optional[float] = 0.8,
        multilingual_reranker_model: str = "rerank-multilingual-v3.0",
        response_synthesizer_model: str = "gpt-4o-2024-08-06",
        response_synthesizer_temperature: float = 0.1,
        response_synthesizer_fallback_model: str = "gpt-4o-2024-08-06",
        response_synthesizer_fallback_temperature: float = 0.1,
    ):
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer()
        self.retrieval = FusionRetrieval(
            vector_store=vector_store,
            top_k=top_k,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            multilingual_reranker_model=multilingual_reranker_model,
        )
        self.response_synthesizer = StreamResponseSynthesizer(
            model=response_synthesizer_model,
            temperature=response_synthesizer_temperature,
            fallback_model=response_synthesizer_fallback_model,
            fallback_temperature=response_synthesizer_fallback_temperature,
        )

    @weave.op()
    async def __call__(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] | None = None,
        reranking: bool = False,
    ) -> RAGPipelineOutput:
        if chat_history is None:
            chat_history = []

        with get_openai_callback() as query_enhancer_cb, Timer() as query_enhancer_tb:
            enhanced_query = await self.query_enhancer(
                {"query": question, "chat_history": chat_history}
            )

        with Timer() as retrieval_tb:
            retrieval_results = await self.retrieval(
                enhanced_query, reranking=reranking
            )

        with get_openai_callback() as response_cb, Timer() as response_tb:
            return await self.response_synthesizer(retrieval_results)
