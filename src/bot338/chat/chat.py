import json
from typing import Any, List, AsyncGenerator

import weave

import wandb
from bot338.chat.config import ChatConfig
from bot338.chat.rag import RAGPipeline, RAGPipelineOutput, StreamRAGPipeline
from bot338.chat.schemas import ChatRequest, ChatResponse, OpenWebUiChatRequest
from bot338.database.schemas import QuestionAnswer
from bot338.retriever import VectorStore
from bot338.utils import Timer, get_logger


logger = get_logger(__name__)


class Chat:
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.
    """

    def __init__(self, vector_store: VectorStore, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.vector_store = vector_store
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.run._label(repo="bot338")

        self.rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            top_k=self.config.top_k,
            search_type=self.config.search_type,
            multilingual_reranker_model=self.config.multilingual_reranker_model,
            response_synthesizer_model=self.config.response_synthesizer_model,
            response_synthesizer_temperature=self.config.response_synthesizer_temperature,
            response_synthesizer_fallback_model=self.config.response_synthesizer_fallback_model,
            response_synthesizer_fallback_temperature=self.config.response_synthesizer_fallback_temperature,
        )

    def _get_answer(
        self, question: str, chat_history: List[QuestionAnswer], reranking: bool = False
    ) -> RAGPipelineOutput:
        history = []
        for item in chat_history:
            history.append(("user", item.question))
            history.append(("assistant", item.answer))

        result = self.rag_pipeline(question, history, reranking)

        return result

    @weave.op()
    def __call__(self, chat_request: ChatRequest) -> ChatResponse:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of `ChatResponse` representing the chat response.
        """
        try:
            result: RAGPipelineOutput = self._get_answer(
                chat_request.question,
                chat_request.chat_history or [],
                chat_request.reranking,
            )

            result_dict = result.model_dump()

            usage_stats = {
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
            result_dict.update({"application": chat_request.application})
            self.run.log(usage_stats)

            return ChatResponse(**result_dict)
        except Exception as e:
            with Timer() as timer:
                result = {
                    "system_prompt": "",
                    "question": chat_request.question,
                    "answer": str(e),
                    "model": "",
                    "sources": "",
                    "source_documents": "",
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            result.update(
                {
                    "time_taken": timer.elapsed,
                    "start_time": timer.start,
                    "end_time": timer.stop,
                }
            )

            return ChatResponse(**result)


class StreamChat(Chat):
    """Class for handling chat interactions.

    Attributes:
        config: An instance of ChatConfig containing configuration settings.
        run: An instance of wandb.Run for logging experiment information.

    TODO
        - vectorstore 안에서 vector_db 로 바뀐점.
        - as_retriever 시, 넘겨주는 인자에 docstore 및, id_key 주의.
    """

    def __init__(self, vector_store: VectorStore, config: ChatConfig):
        """Initializes the Chat instance.

        Args:
            config: An instance of ChatConfig containing configuration settings.
        """
        self.vector_store = vector_store
        self.config = config
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            job_type="chat",
        )
        self.run._label(repo="bot338")

        self.rag_pipeline = StreamRAGPipeline(
            vector_store=vector_store,
            top_k=self.config.top_k,
            search_type=self.config.search_type,
            fetch_k=self.config.fetch_k,
            lambda_mult=self.config.lambda_mult,
            multilingual_reranker_model=self.config.multilingual_reranker_model,
            response_synthesizer_model=self.config.response_synthesizer_model,
            response_synthesizer_temperature=self.config.response_synthesizer_temperature,
            response_synthesizer_fallback_model=self.config.response_synthesizer_fallback_model,
            response_synthesizer_fallback_temperature=self.config.response_synthesizer_fallback_temperature,
        )

    async def _get_answer(
        self, question: str, chat_history: List[QuestionAnswer], reranking: bool = False
    ) -> AsyncGenerator[dict, None]:
        history = []
        for item in chat_history:
            history.append(("user", item.question))
            history.append(("assistant", item.answer))

        return await self.rag_pipeline(question, history, reranking)

    async def _completion(
        self, query: str, chat_history: List[dict[str, Any]], reranking: bool = False
    ) -> AsyncGenerator[dict, None]:
        history = []
        for message in chat_history:
            role = message["role"]
            content = message["content"]

            if role in ["user", "assistant"]:
                history.append((role, content))
        return await self.rag_pipeline(query, history, reranking)

    async def __call__(
        self, chat_request: OpenWebUiChatRequest
    ) -> AsyncGenerator[dict, None]:
        """Handles the chat request and returns the chat response.

        Args:
            chat_request: An instance of ChatRequest representing the chat request.

        Returns:
            An instance of `ChatResponse` representing the chat response.
        """

        chain: AsyncGenerator[dict, None] = await self._completion(
            chat_request.query,
            chat_request.chat_history or [],
            chat_request.reranking,
        )

        async for chunk in chain:
            if isinstance(chunk, dict):
                if "response" in chunk:
                    yield chunk["response"]
                else:
                    yield ""
            else:
                yield ""
