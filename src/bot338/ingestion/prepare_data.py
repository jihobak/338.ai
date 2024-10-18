import json
import pathlib
from multiprocessing import Pool, cpu_count
from typing import AsyncIterator, Iterator, Dict, Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import wandb

from bot338.ingestion.config import DataSource, DataStoreConfig, BillsStoreConfig
from bot338.ingestion.utils import fetch_git_repo
from bot338.utils import get_logger


logger = get_logger(__name__)


class DataLoader(BaseLoader):
    """A base class for data loaders.

    This class provides a base implementation for lazy loading of documents.
    Subclasses should implement the `lazy_load` method to define the specific
    loading behavior.
    """

    def __init__(self, config: DataStoreConfig):
        """Initializes the DataLoader instance.

        Args:
            config: The configuration for the data store.
        """
        self.config: DataStoreConfig = config
        self.metadata: Dict[str, Any] = {}

    def load(self) -> list[Document]:
        """Loads the documents.

        Returns:
            A list of Document objects.
        """
        documents = list(self.lazy_load())
        self.metadata.update({"num_documents": len(documents)})
        return documents

    def _get_local_paths(self) -> list[pathlib.PosixPath]:
        if self.config.data_source.is_git_repo:
            self.metadata = fetch_git_repo(
                self.config.data_source, self.config.data_source.git_id_file
            )

        local_paths = []
        for file_pattern in self.config.data_source.file_patterns:
            local_paths.extend(
                list(
                    (
                        self.config.data_source.local_path
                        / self.config.data_source.base_path
                    ).rglob(file_pattern)
                )
            )
        return local_paths


class BillsDataLoader(DataLoader):

    @staticmethod
    def read_json_file(path: pathlib.PosixPath):
        return json.load(path.open("r"))

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""
        local_paths = self._get_local_paths()

        for path in local_paths:
            try:
                raw_data = self.read_json_file(path)
                # raw_data keys
                # ['summary', 'openai_large3_desc', 'metadata']
                metadata = raw_data["metadata"]
                metadata["source_type"] = self.config.source_type  # bill

                doc = Document(page_content=raw_data["summary"], metadata=metadata)
            except Exception as e:
                logger.warning(f"Failed to load documentation {path} due to {e}")
            else:
                yield doc


SOURCE_TYPE_TO_LOADER_MAP = {
    "bills": BillsDataLoader,
}


def get_loader_from_config(config: DataStoreConfig) -> DataLoader:
    """Get the DataLoader class based on the source type.

    Args:
        config: The configuration for the data store.

    Returns:
        The DataLoader class.
    """
    source_type: str = config.source_type

    return SOURCE_TYPE_TO_LOADER_MAP[source_type](config)


def load_from_config(config: DataStoreConfig) -> pathlib.Path:
    """
    1. config -> loader 를 만든다.
    2. config.docstore_dir 까지의 path 를 만든다.
    3. docstore_dir 에 config 를 기록한다.
    4. loader 를 통해서 데이터를 불러와서 documnets.jsonl 로 저장한다.
    5. loader 의 메타데이터를 저장한다.
    """
    loader = get_loader_from_config(config)
    loader.config.docstore_dir.mkdir(parents=True, exist_ok=True)

    with (loader.config.docstore_dir / "config.json").open("w") as f:
        f.write(loader.config.model_dump_json())

    with (loader.config.docstore_dir / "documents.jsonl").open("w") as f:
        for document in loader.load():
            document_json = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            f.write(json.dumps(document_json) + "\n")

    with (loader.config.docstore_dir / "metadata.json").open("w") as f:
        json.dump(loader.metadata, f)

    return loader.config.docstore_dir


def load(
    project: str,
    entity: str,
    raw_data_path: pathlib.PosixPath,
    result_artifact_name: str = "raw_dataset",
) -> str:
    """Load and prepare data for the Wandbot ingestion system.

    This function initializes a Wandb run, creates an artifact for the prepared dataset,
    and loads and prepares data from different loaders. The prepared data is then saved
    in the docstore directory and added to the artifact.

    Args:
        project: The name of the Wandb project.
        entity: The name of the Wandb entity.
        result_artifact_name: The name of the result artifact. Default is "raw_dataset".

    Returns:
        The latest version of the prepared dataset artifact in the format
        "{entity}/{project}/{result_artifact_name}:latest".
    """
    run = wandb.init(project=project, entity=entity, job_type="prepare_dataset")
    artifact = wandb.Artifact(
        result_artifact_name,
        type="dataset",
        description="Raw documents for bot338",
    )

    data_source = DataSource(
        file_patterns=["*.json"],
        local_path=raw_data_path,
        base_path="bills",
    )
    bills_store_config = BillsStoreConfig(data_source=data_source)

    configs = [bills_store_config]

    pool = Pool(cpu_count() - 1)
    results = pool.imap_unordered(load_from_config, configs)

    for docstore_path in results:
        artifact.add_dir(
            str(docstore_path),
            name=docstore_path.name,
        )
    run.log_artifact(artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


if __name__ == "__main__":
    ...
