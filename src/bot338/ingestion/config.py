import datetime
import os
import pathlib
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bot338.utils import get_logger

logger = get_logger(__name__)


class DataSource(BaseSettings):
    """
    - local_path: 모든 데이터 검색이 시작되는 루트 디렉토리
    - base_path: 루트 디렉토리에서 목표로하는 구체적인 시작 디렉토리
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    cache_dir: pathlib.Path = Field(
        pathlib.Path("cache"), validation_alias="BOT338_CACHE_DIR"
    )
    ignore_cache: bool = False
    remote_path: str = ""
    repo_path: str = ""
    local_path: Optional[pathlib.Path] = None
    branch: Optional[str] = None
    base_path: Optional[str] = ""
    file_patterns: List[str] = ["*.*"]
    is_git_repo: bool = False
    git_id_file: Optional[pathlib.Path] = Field(None)

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v):
        if isinstance(v, str):
            return pathlib.Path(v)
        return v


class DataStoreConfig(BaseModel):
    name: str = "docstore"
    source_type: str = ""
    data_source: DataSource  # = DataSource()
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


def create_data_source() -> DataSource:
    return DataSource(
        file_patterns=["*.json"],
        local_path=pathlib.Path("raw_data/"),
        base_path="dev_bill",
    )


class BillStoreConfig(DataStoreConfig):
    """
    - default_factory 가 없다면, import BillStoreConfig 하는 순간, DataSource 가
    초기화되기 때문에, 환경변수 로딩전에 초기화가되어 문제가 발생 할 수 있다.
    """

    name: str = "dev"
    source_type: str = "bill"
    docstore_dir: pathlib.Path = pathlib.Path("bill")
    data_source: DataSource = Field(default_factory=create_data_source)


class VectorStoreConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    collection_name: str = "vectorstore"
    persist_dir: pathlib.Path = Field(
        "vectorstore", validation_alias="BOT338_VECTSTORE_PERSIST_DIR"
    )
    embedding_model_name: str = Field(
        "text-embedding-3-small", validation_alias="INDEXING_EMBEDDING_MODEL"
    )
    batch_size: int = 256  # 임베딩 처리 batch size

    artifact_url: str = Field(
        f"{os.environ.get('WANDB_ENTITY', 'trendfollowing')}/{os.environ.get('WANDB_PROJECT', '338lab')}/bot338_index:latest",
        validation_alias="WANDB_INDEX_ARTIFACT",
    )

    # parent document(full docs) 를 저장하기 위한 용도 docstore
    docstore_uri: str = Field(
        "mongodb://localhost:27017", validation_alias="DOCSTORE_URI"
    )
    docstore_db_name: str = Field("bot338", validation_alias="DOCSTORE_DB_NAME")
    docstore_collection_name: str = Field(
        "collection_name", validation_alias="DOCSTORE_COLLECTION"
    )
    id_key: str = "parent_id"
