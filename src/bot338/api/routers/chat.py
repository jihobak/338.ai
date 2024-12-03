from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette import status

from bot338.chat.chat import Chat, ChatConfig, StreamChat
from bot338.chat.schemas import ChatRequest, ChatResponse, OpenWebUiChatRequest
from bot338.utils import get_logger


logger = get_logger(__name__)


chat_config = ChatConfig()
logger.info(f"Chat config: {chat_config}")
# chat: Chat | None = None
chat: StreamChat | None = None

router = APIRouter(prefix="/chat", tags=["chat"])


class APIQueryRequest(ChatRequest):
    pass


class APIQueryResponse(ChatResponse):
    pass


class APICompletionRequest(OpenWebUiChatRequest):
    pass


class ChatStreamError(Exception):
    """A custom exception for handling chat stream errors with status code and message."""

    def __init__(self, status_code: int, message: str):
        """
        Initializes a ChatStreamError with a status code and an error message.

        Args:
            status_code (int): The HTTP status code to be returned.
            message (str): The error message to be sent to the client.
        """
        self.status_code = status_code  # HTTP 상태 코드
        self.message = message  # 에러 메시지
        super().__init__(message)

    def __str__(self):
        """Returns a formatted error message with the status code."""
        return f"{self.status_code} Error: {self.message}"


@router.post("/query", response_model=APIQueryResponse, status_code=status.HTTP_200_OK)
def query(
    request: APIQueryRequest,
) -> APIQueryResponse:
    """Executes a query using the chat function and returns the result as an APIQueryResponse.

    Args:
        request: The APIQueryRequest object containing the question and chat history

    Returns:
        The APIQueryResponse object containing the result of the query.
    """

    result: ChatResponse = chat(
        ChatRequest(
            question=request.question,
            chat_history=request.chat_history,
            application=request.application,
            reranking=request.reranking,
        )
    )

    result = APIQueryResponse(**result.model_dump())

    return result


@router.post("/completion", status_code=status.HTTP_200_OK)
async def completion(
    request: APICompletionRequest,
) -> StreamingResponse:
    """요청에 대한 채팅 스트림을 생성하고 반환하는 엔드포인트"""

    try:
        chat_stream: AsyncGenerator[dict, None] = chat(
            OpenWebUiChatRequest(
                query=request.query,
                chat_history=request.chat_history,
                application=request.application,
                reranking=request.reranking,
            )
        )

        return StreamingResponse(chat_stream, media_type="text/event-stream")

    except BaseException as e:
        # 기타 예상치 못한 에러를 500 상태 코드로 처리
        raise ChatStreamError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Unexpected error: {str(e)}",
        ) from e
