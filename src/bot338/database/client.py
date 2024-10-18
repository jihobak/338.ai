"""This module provides Database and DatabaseClient class for managing database operations.

The Database class provides a connection to the database and manages the session. It also provides methods for
getting and setting the current session object and the name of the database.

The DatabaseClient class uses an instance of the Database class to perform operations such as getting and creating
chat threads, question answers, and feedback from the databases.

Typical usage example:

    db_client = DatabaseClient()
    chat_thread = db_client.get_chat_thread(application='app1', thread_id='123')
    question_answer = db_client.create_question_answer(question_answer=QuestionAnswerCreateSchema())
"""

import json
from typing import Any, List

from sqlalchemy.future import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from bot338.database.config import DataBaseConfig
from bot338.database.models import ChatThread as ChatThreadModel
from bot338.database.models import FeedBack as FeedBackModel
from bot338.database.models import QuestionAnswer as QuestionAnswerModel
from bot338.database.schemas import (
    ChatThreadCreate as ChatThreadCreateSchema,
    FeedbackCreate,
)
from bot338.database.schemas import Feedback as FeedbackSchema
from bot338.database.schemas import (
    QuestionAnswerCreate as QuestionAnswerCreateSchema,
)
from bot338.utils import get_logger


logger = get_logger(__name__)


class Database:
    """A class representing a database connection.

    This class provides a connection to the database and manages the session.

    Attributes:
        db_config: An instance of the DataBaseConfig class.
        SessionLocal: A sessionmaker object for creating sessions.
        db: The current session object.
        name: The name of the database.
    """

    db_config: DataBaseConfig = DataBaseConfig()

    def __init__(self, database: str | None = None):
        """Initializes the Database instance.

        Args:
            database: The URL of the database. If None, the default URL is used.
        """
        if database is not None:
            engine: Engine = create_engine(
                url=database, connect_args=self.db_config.connect_args
            )
        else:
            engine: Engine = create_engine(
                url=self.db_config.SQLALCHEMY_DATABASE_URL,
                connect_args=self.db_config.connect_args,
            )
        self.SessionLocal: Any = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

    def __get__(self, instance, owner) -> Any:
        """Gets the current session object

        Args:
            instance: The instance of the owner class.
            owner: The owner class.

        Returns:
            The current session object.
        """
        if not hasattr(self, "db"):
            self.db: Any = self.SessionLocal()

        return self.db

    def __set__(self, instance, value) -> None:
        """Sets the current session object.

        Args:
            instance: The instance of the owner class.
            value: The new session object.
        """
        self.db = value

    def __set_name__(self, owner, name) -> None:
        """Sets the name of the database.

        Args:
            owner: The owner class.
            name: The name of the database.
        """
        self.name: str = name


class DatabaseClient:
    database: Database = Database()

    def __init__(self, database: str | None = None):
        """Initializes the DatabaseClient instance.

        Args:
            database: The URL of the database. If None, the default URL is used.
        """
        if database is not None:
            self.database = Database(database=database)

    def get_chat_thread(
        self, application: str, thread_id: str
    ) -> ChatThreadModel | None:
        """Gets a chat thread from the database.

        Args:
            application: The application name.
            thread_id: The ID of the chat thread.

        Returns:
            The chat thread model if found, None otherwise.
        """
        chat_thread: ChatThreadModel | None = (
            self.database.query(ChatThreadModel)
            .filter(
                ChatThreadModel.thread_id == thread_id,
                ChatThreadModel.application == application,
            )
            .first()
        )

        return chat_thread

    def create_chat_thread(
        self, chat_thread: ChatThreadCreateSchema
    ) -> ChatThreadModel:
        """Creates a chat thread in the database.

        Args:
            chat_thread: The chat thread to create.

        Returns:
            The created chat thread model.
        """
        try:
            chat_thread: ChatThreadModel = ChatThreadModel(
                thread_id=chat_thread.thread_id,
                application=chat_thread.application,
            )
            self.database.add(chat_thread)
            self.database.flush()
            self.database.commit()
            self.database.refresh(chat_thread)
        except Exception as e:
            logger.error(f"Create chat thread failed with error: {e}")
            self.database.rollback()

        return

    def get_question_answer(
        self,
        question_answer_id: str,
        thread_id: str,
    ) -> QuestionAnswerModel | None:
        """Gets a question answer from the database.

        Args:
            question_answer_id: The ID of the question answer.
            thread_id: The ID of the chat thread.

        Returns:
            The question answer model if found, None otherwise.
        """
        question_answer: QuestionAnswerModel | None = (
            self.database.query(QuestionAnswerModel)
            .filter(
                QuestionAnswerModel.thread_id == thread_id,
                QuestionAnswerModel.question_answer_id == question_answer_id,
            )
            .first()
        )
        return question_answer

    def create_question_answer(
        self, question_answer: QuestionAnswerCreateSchema
    ) -> QuestionAnswerModel:
        """Creates a question answer in the database.

        Args:
            question_answer: The question answer to create.

        Returns:
            The created question answer model.
        """
        try:
            question_answer: QuestionAnswerModel = QuestionAnswerModel(
                **question_answer.model_dump()
            )
            self.database.add(question_answer)
            self.database.flush()
            self.database.commit()
            self.database.refresh(question_answer)
        except Exception as e:
            logger.error(f"Create question answer failed with error: {e}")
            self.database.rollback()
        return question_answer

    def get_feedback(self, question_answer_id: str) -> FeedBackModel | None:
        """Gets feedback from the database.

        Args:
            question_answer_id: The ID of the question answer.

        Returns:
            The feed model if found, None otherwise.
        """
        feedback: FeedBackModel | None = (
            self.database.query(FeedBackModel)
            .filter(FeedBackModel.question_answer_id == question_answer_id)
            .first()
        )
        return feedback

    def create_feedback(self, feedback: FeedbackCreate) -> FeedBackModel:
        """Creates feedback in the database.

        # TODO
        - FeedbackSchema -> FeedbackCreate 로 변경하는게 맞다고 생각

        Args:
            feedback: The feedback to create.

        Returns:
            The created feedback model.
        """
        if feedback.rating:
            feedback: FeedBackModel = FeedBackModel(**feedback.model_dump())
            try:
                self.database.add(feedback)
                self.database.flush()
                self.database.commit()
                self.database.refresh(feedback)
            except Exception as e:
                logger.error(f"Create feedback fail with error: {e}")
                self.database.rollback()

            return feedback

    def get_all_question_answers(self, time: Any = None) -> List[dict[str, Any]]:
        question_answers = self.database.query(QuestionAnswerModel)
        if time is not None:
            question_answers = question_answers.filter(
                QuestionAnswerModel.end_time >= time
            )
        question_answers = question_answers.all()
        if question_answers is not None:
            question_answers = [
                json.loads(
                    QuestionAnswerCreateSchema.model_validate(
                        question_answer
                    ).model_dump_json()
                )
                for question_answer in question_answers
            ]
            return question_answers
