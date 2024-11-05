import json
import os
import pathlib
from typing import List

from bot338.ingestion.report import create_ingestion_report
from dotenv import load_dotenv, find_dotenv
from pydantic import Field
import wandb

from bot338.ingestion import prepare_data, preprocess_data, vectorstores
from bot338.ingestion.config import BillStoreConfig, VectorStoreConfig
from bot338.utils import get_logger


logger = get_logger(__name__)


def main():
    """
    - poetry run python -m src.bot338.ingestion
    """
    env_file_path = find_dotenv()
    load_env = load_dotenv(env_file_path)

    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    logger.info(f"[WANDB] {project=}, {entity=}")

    current_path = pathlib.Path(__file__).resolve()
    ingestion_folder = current_path.parent  # ingestion
    src_folder = ingestion_folder.parent.parent  # bot338
    root_folder = src_folder.parent  # src

    # bill_config = BillStoreConfig()
    # logger.info(f"{bill_config.docstore_dir}")
    result_artifact_name = "dev"
    raw_artifact = prepare_data.load(project, entity, result_artifact_name)
    # raw_artifact='trendfollowing/338lab/dev:latest'

    # ---
    preprocessed_artifact = preprocess_data.load(
        project, entity, raw_artifact, f"transformed_{result_artifact_name}"
    )
    # preprocessed_artifact='trendfollowing/338lab/transformed_dev:latest'

    vectorstore_artifact = vectorstores.load(
        project, entity, preprocessed_artifact, "bot338_dev_index"
    )
    # vectorstore_artifact='trendfollowing/338lab/bot338_index:latest'

    # create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)


if __name__ == "__main__":
    main()  # 108
