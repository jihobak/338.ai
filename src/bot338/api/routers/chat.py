from fastapi import APIRouter
from starlette import status

from bot338.chat.chat import Chat, ChatConfig
from bot338.chat.schemas import ChatRequest, ChatResponse
from bot338.utils import get_logger


logger = get_logger(__name__)


chat_config = ChatConfig()
logger.info(f"Chat config: {chat_config}")
chat: Chat | None = None


router = APIRouter(prefix="/chat", tags=["chat"])


class APIQueryRequest(ChatRequest):
    pass


class APIQueryResponse(ChatResponse):
    pass


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
