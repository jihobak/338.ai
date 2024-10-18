import json
import os
import pathlib
from typing import List

from bot338.ingestion.report import create_ingestion_report
from dotenv import load_dotenv, find_dotenv
import wandb

from bot338.ingestion import prepare_data, preprocess_data, vectorstores
from bot338.utils import get_logger


logger = get_logger(__name__)


def main2():
    a = find_dotenv()
    b = load_dotenv(a)

    logger.info(f"{a=}")
    logger.info(f"{b=}")


def main():
    _ = load_dotenv(find_dotenv())
    project = os.environ.get("WANDB_PROJECT", "338lab")
    entity = os.environ.get("WANDB_ENTITY", "trendfollowing")
    logger.info(f"{project=}, {entity=}")

    current_path = pathlib.Path(__file__).resolve()  # ingestion
    ingestion_folder = current_path.parent
    src_folder = ingestion_folder.parent.parent
    root_folder = src_folder.parent
    raw_data_path = root_folder / "raw_data"
    logger.info(f"{raw_data_path=}")
    # raw_data_path=PosixPath('/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/raw_data')

    result_artifact_name = "dojo_mode"
    raw_artifact = prepare_data.load(
        project, entity, raw_data_path, result_artifact_name
    )
    logger.info(f"{raw_artifact=}")
    # raw_artifact='trendfollowing/338lab/dojo_mode:latest'

    preprocessed_artifact = preprocess_data.load(
        project, entity, raw_artifact, "transformed_data"
    )
    logger.info(f"{preprocessed_artifact=}")
    # preprocessed_artifact='trendfollowing/338lab/transformed_data:latest'

    vectorstore_artifact = vectorstores.load(
        project, entity, preprocessed_artifact, "bot338_index"
    )
    logger.info(f"{vectorstore_artifact=}")
    # INFO : vectorstore_artifact='trendfollowing/338lab/bot338_index:latest'

    create_ingestion_report(project, entity, raw_artifact, vectorstore_artifact)
    logger.info(vectorstore_artifact)


if __name__ == "__main__":
    main()  # 108
