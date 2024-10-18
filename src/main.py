import json
import os
import pathlib
from typing import List

from dotenv import load_dotenv, find_dotenv
import wandb
from langchain_core.documents import Document
from bot338.ingestion import preprocess_data
from bot338.utils import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    project = os.environ.get("WANDB_PROJECT", "338lab")
    entity = os.environ.get("WANDB_ENTITY", "trendfollowing")
    logger.info(f"{project=}, {entity=}")

    import wandb

    run = wandb.init(project=project, entity=entity)
    artifact = run.use_artifact(
        "trendfollowing/338lab/transformed_data:v0", type="dataset"
    )
    artifact_dir = artifact.download()

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    transformed_documents = []
    for document_file in document_files:
        logger.info(f"{document_file=}")
        #  document_file=PosixPath('/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/artifacts/transformed_data:v0/bills/documents.jsonl')
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))
