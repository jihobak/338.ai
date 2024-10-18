import datetime
import os
import pathlib
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bot338.utils import get_logger

logger = get_logger(__name__)


class DataSource(BaseSettings):
    cache_dir: pathlib.Path = Field("data/cache/raw_data", env="BOT338_CACHE_DIR")
    ignore_cache: bool = False
    remote_path: str = ""
    repo_path: str = ""
    local_path: Optional[pathlib.Path] = None
    branch: Optional[str] = None
    base_path: Optional[str] = ""
    file_patterns: List[str] = ["*.*"]
    is_git_repo: bool = False
    git_id_file: Optional[pathlib.Path] = Field(None, env="BOT338_GIT_ID_FILE")


class DataStoreConfig(BaseModel):
    name: str = "docstore"
    source_type: str = ""
    data_source: DataSource = DataSource()
    docstore_dir: pathlib.Path = pathlib.Path("docstore")

    @model_validator(mode="after")
    @classmethod
    def _set_cache_paths(cls, values: "DataStoreConfig") -> "DataStoreConfig":
        values.docstore_dir = (
            values.data_source.cache_dir
            / "_".join(values.name.split())
            / values.docstore_dir
        )
        data_source = values.data_source

        if data_source.repo_path:
            data_source.is_git_repo = (
                urlparse(data_source.repo_path).netloc == "github.com"
            )
            local_path = urlparse(data_source.repo_path).path.split("/")[-1]
            if not data_source.local_path:
                data_source.local_path = (
                    data_source.cache_dir / "_".join(values.name.split()) / local_path
                )
            if data_source.is_git_repo:
                if data_source.git_id_file is None:
                    logger.debug(
                        "The source data is a git repo but no git_id_file is set."
                        " Attempting to use the default ssh id file"
                    )
                    data_source.git_id_file = pathlib.Path.home() / ".ssh" / "id_rsa"
        values.data_source = data_source

        return values


class BillsStoreConfig(DataStoreConfig):
    name: str = "bills from assembly"
    source_type: str = "bill"
    docstore_dir: pathlib.Path = pathlib.Path("bills")


class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    collection_name: str = "vectorstore"
    persist_dir: pathlib.Path = Field(
        "data/cache/vectorstore", env="BOT338_VECTSTORE_PERSIST_DIR"
    )
    embedding_model_name: str = "text-embedding-3-large"
    tokenizer_model_name: str = "text-embedding-3-large"
    batch_size: int = 256  # 임베딩 처리 batch size

    # TODO
    # lancedb_index 대신에 bot338_index 를 사용해야 할 수 있다.
    artifact_url: str = Field(
        f"{os.environ.get('WANDB_ENTITY', 'trendfollowing')}/{os.environ.get('WANDB_PROJECT', '338lab')}/lancedb_index:latest",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
