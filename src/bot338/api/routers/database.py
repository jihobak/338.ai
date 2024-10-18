from fastapi import APIRouter
from starlette import status
from starlette.responses import Response

import wandb
from bot338.database.client import DatabaseClient
from bot338.database.schemas import (
    ChatThread,
    ChatThreadCreate,
    Feedback,
    FeedbackCreate,
    QuestionAnswer,
    QuestionAnswerCreate,
)
from bot338.database.models import ChatThread as ChatThreadModel
from bot338.database.models import QuestionAnswer as QuestionAnswerModel
from bot338.utils import get_logger

logger = get_logger(__name__)


db_client: DatabaseClient | None = None


router = APIRouter(
    prefix="/data",
    tags=["database", "crud"],
)


class APIQuestionAnswerRequest(QuestionAnswerCreate):
    pass


class APIQuestionAnswerResponse(QuestionAnswer):
    pass


@router.post(
    "/question_answer",
    response_model=APIQuestionAnswerResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_question_answer(
    request: APIQuestionAnswerRequest, response: Response
) -> APIQuestionAnswerResponse:
    """Creates a question answer

    Args:
        request: The request object containing the question answer data.
        response: The response object to update with the result.


    Returns:
        The created question answer or None if creation failed.
    """
    question_answer: QuestionAnswerModel = db_client.create_question_answer(request)
    if question_answer is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    return question_answer


class APIGetChatThreadRequest(ChatThreadCreate):
    pass


class APIGetChatThreadResponse(ChatThread):
    pass


class APICreateChatThreadRequest(ChatThreadCreate):
    pass


@router.get(
    "/chat_thread/{application}/{thread_id}",
    response_model=APIGetChatThreadResponse,
    status_code=status.HTTP_200_OK,
)
def get_chat_thread(
    application: str, thread_id: str, response: Response
) -> APIGetChatThreadResponse:
    """Retrieves a chat thread from the database.

    Args:
        application: The application name
        thread_id: The ID of the chat thread.
        response: The HTTP response object.

    Return:
        The retrieved or created chat thread.
    """
    chat_thread: ChatThreadModel = db_client.get_chat_thread(
        application=application, thread_id=thread_id
    )

    if chat_thread is None:
        chat_thread = db_client.create_chat_thread(
            APICreateChatThreadRequest(application=application, thread_id=thread_id)
        )
        response.status_code = status.HTTP_201_CREATED

    if chat_thread is None:
        response.status_code = status.HTTP_400_BAD_REQUEST

    return chat_thread


class APIFeedbackRequest(FeedbackCreate):
    pass


class APIFeedbackResponse(Feedback):
    pass


@router.post(
    "/feedback",
    response_model=APIFeedbackResponse | None,
    status_code=status.HTTP_201_CREATED,
)
def feedback(request: APIFeedbackRequest, response: Response) -> APIFeedbackResponse:
    """Handles the feedback request and logs the feedback data.


    Args:
        request: The feedback request object.
        response: The response object.


    Returns:
        The feedback response object.
    """
    feedback_response = db_client.create_feedback(request)
    if feedback_response is not None:
        wandb.log(
            {
                "feedback": wandb.Table(
                    columns=list(request.model_dump().keys()),
                    data=[list(request.model_dump().values())],
                )
            }
        )
    else:
        response.status_code = status.HTTP_400_BAD_REQUEST

    return feedback_response
