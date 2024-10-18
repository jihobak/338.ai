import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import dotenv
from dotenv import find_dotenv
import pandas as pd
import weave
from fastapi import BackgroundTasks, FastAPI

import wandb
from bot338.api.routers import chat as chat_router
from bot338.api.routers import database as database_router
from bot338.api.routers import retrieve as retrieve_router
from bot338.chat.chat import ChatConfig
from bot338.database.database import engine
from bot338.database.models import Base
from bot338.ingestion.config import VectorStoreConfig
from bot338.retriever import VectorStore
from bot338.utils import get_logger

logger = get_logger(__name__)
last_backup = datetime.now().astimezone(timezone.utc)

dotenv_path = os.path.join(os.path.dirname(__file__), "../../../.env")
logger.info(f"loading env variables .... : {dotenv.load_dotenv(dotenv_path)}")

# turn off chromadb telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

weave.init(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}")

is_initialized = False


async def initialize():
    """
    TODO
        - dependency injection 으로 해결가능한 부분
            - chat_router.chat = ...
            - database_router.db_client = ...
            - retrieve_router.retriever = ...
    """
    logger.info(f"Initializing bot338")
    global is_initialized
    if not is_initialized:
        vector_store = VectorStore.from_config(VectorStoreConfig())
        chat_config = ChatConfig()
        chat_router.chat = chat_router.Chat(
            vector_store=vector_store, config=chat_config
        )
        logger.info(f"Initialized chat router")
        database_router.db_client = database_router.DatabaseClient()
        logger.info(f"Initialized database client")

        retrieve_router.retriever = retrieve_router.SimpleRetrievalEngine(
            vector_store=vector_store,
            rerank_models={
                "multilingual_reranker_model": chat_config.multilingual_reranker_model,
            },
        )
        logger.info(f"Initialized retrieve router")
        logger.info(f"bot338 initialization complete")
        is_initialized = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the lifespan of the application.

    This function is called by the Uvicorn server to handle the lifespan of the application.
    It is used to perform any necessary startup and shutdown operations.

    Returns:
        None
    """

    Base.metadata.create_all(bind=engine)

    if os.getenv("BOT338_EVALUATION", False):
        logger.info("Lifespan starting, initializing wandbot for evaluation mode.")
        await initialize()

    async def backup_db():
        """Periodically backs up the database to a table.

        This function runs periodically and retrieves all question-answer threads from the database since the last backup.
        It then creates a pandas DataFrame from the retrieved threads and logs it to a table using Weights & Biases.
        The last backup timestamp is updated after each backup.

        Returns:
            None
        """
        global last_backup
        while True:
            chat_threads = database_router.db_client.get_all_question_answers(
                last_backup
            )
            if chat_threads is not None:
                chat_table = pd.DataFrame([chat_thread for chat_thread in chat_threads])
                last_backup = datetime.now().astimezone(timezone.utc)
                logger.info(
                    f"Backing up database to Table at {last_backup}: Number of chat threads: {len(chat_table)}"
                )
                wandb.log({"question_answers_db": wandb.Table(dataframe=chat_table)})
            await asyncio.sleep(600)

    _ = asyncio.create_task(backup_db())
    yield
    if wandb.run is not None:
        wandb.run.finish()


app = FastAPI(title="Bot338", name="bot338", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root(background_tasks: BackgroundTasks):
    logger.info("Received request to root endpoint")
    background_tasks.add_task(initialize)
    logger.info("Added initialize task to background tasks")
    return {"message": "Initialization started in the background"}


app.include_router(chat_router.router)
app.include_router(database_router.router)
app.include_router(retrieve_router.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
