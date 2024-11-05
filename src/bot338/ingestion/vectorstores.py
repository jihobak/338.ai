"""This module contains functions for loading and managing vector stores in the Bot338 ingestion system.

The module includes the following functions:
- `load`: Loads the vector store from the specified source artifact path and returns the name of the resulting artifact.

Typical usage example:

    project = "bot338-dev"
    entity = "bot338"
    source_artifact_path = "bot338/bot338-dev/raw_dataset:latest"
    result_artifact_name = "bot338_index"
    load(project, entity, source_artifact_path, result_artifact_name)
"""

import json
import pathlib
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from tqdm import trange

import wandb
from bot338.ingestion.config import VectorStoreConfig
from bot338.utils import get_logger

logger = get_logger(__name__)


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "bot338_index",
) -> str:
    """Load the vector store.

    Loads the vector store from the specified source artifact path and returns the name of the resulting artifact.

    Args:
        project: The name of the project.
        entity: The name of the entity.
        source_artifact_path: The path to the source artifact.
        result_artifact_name: The name of the resulting artifact. Defaults to "bot338_index".

    Returns:
        The name of the resulting artifact.

    Raises:
        wandb.Error: An error occurred during the loading process.
    """
    config: VectorStoreConfig = VectorStoreConfig()
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="create_vectorstore"
    )
    artifact: wandb.Artifact = run.use_artifact(source_artifact_path, type="dataset")
    artifact_dir: str = artifact.download()
    # artifact_dir='/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/artifacts/transformed_dev:v3'

    embedding_fn = OpenAIEmbeddings(
        model=config.embedding_model_name,
    )
    vectorstore_dir = config.persist_dir
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    # vectorstore_dir=PosixPath('data/cache/vectorstore')

    document_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    transformed_documents = []
    for document_file in document_files:
        #  document_file=PosixPath('/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/artifacts/transformed_data:v0/bills/documents.jsonl')
        with document_file.open() as f:
            for line in f:
                transformed_documents.append(Document(**json.loads(line)))

    # 기본 테이블 네임은 table_name: Optional[str] = "vectorstore",
    vector_db = LanceDB(
        uri=str(config.persist_dir),
        table_name=config.collection_name,
        embedding=embedding_fn,
        mode="overwrite",
    )

    for batch_idx in trange(0, len(transformed_documents), config.batch_size):
        batch = transformed_documents[batch_idx : batch_idx + config.batch_size]
        vector_db.add_documents(batch)

    result_artifact = wandb.Artifact(
        name=result_artifact_name,
        type="vectorstore",
    )
    # result_artifact=<Artifact bot338_dev_index>

    result_artifact.add_dir(
        local_path=str(config.persist_dir),
    )
    run.log_artifact(result_artifact, aliases=["lancedb_index", "latest"])

    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"
