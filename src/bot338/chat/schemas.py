from datetime import datetime
from typing import List

from pydantic import BaseModel

from bot338.database.schemas import QuestionAnswer


class ChatThreadBase(BaseModel):
    question_answers: list[QuestionAnswer] | None = []


class ChatThreadCreate(ChatThreadBase):
    thread_id: str
    application: str

    class Config:
        use_enum_values = True


class ChatThread(ChatThreadCreate):
    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    question: str
    chat_history: List[QuestionAnswer] | None = None
    application: str | None = None
    reranking: bool = False


class ChatResponse(BaseModel):
    system_prompt: str
    question: str
    answer: str
    model: str
    source_documents: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    time_taken: float
    start_time: datetime
    end_time: datetime
    sources: str = ""
    api_call_statuses: dict = {}
