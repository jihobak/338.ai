"""This module contains the configuration settings for Bot338.

The `ChatConfig` class in this module is used to define various settings for Bot338, such as the model name, 
maximum retries, fallback model name, chat temperature, chat prompt, index artifact, embeddings cache, verbosity, 
wandb project and entity, inclusion of sources, and query tokens threshold. These settings are used throughout the 
chatbot's operation to control its behavior.

Typical usage example:

  from Bot338.chat.config import ChatConfig
  config = ChatConfig()
  print(config.chat_model_name)
"""

import os
import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    wandb_project: str | None = Field("338lab", validation_alias="WANDB_PROJECT")
    wandb_entity: str | None = Field("trendfollowing", validation_alias="WANDB_ENTITY")
    index_artifact: str = Field(
        f"{os.environ.get('WANDB_ENTITY', 'trendfollowing')}/{os.environ.get('WANDB_PROJECT', '338lab')}/bot338_index:latest",
        validation_alias="WANDB_INDEX_ARTIFACT",
    )
    # Retrieval settings
    top_k: int = Field(5, validation_alias="TOP_K")
    search_type: str = Field("mmr", validation_alias="SEARCH_TYPE")
    fetch_k: int = Field(10, validation_alias="FETCH_K")
    lambda_mult: float | None = Field(0.5, validation_alias="LAMBDA_MULT")

    # Cohere reranker models
    multilingual_reranker_model: str = Field(
        "rerank-multilingual-v3.0", validation_alias="RERANKER_MODEL"
    )
    # Response synthesis settings
    response_synthesizer_model: str = Field(
        "gpt-4o-2024-08-06", validation_alias="SYNTHESIZER_MODEL"
    )
    response_synthesizer_temperature: float = Field(
        0.1, validation_alias="SYNTHESIZER_TEMP"
    )
    response_synthesizer_fallback_model: str = Field(
        "gpt-4o-2024-08-06", validation_alias="SYNTHESIZER_FALLBACK_MODEL"
    )
    response_synthesizer_fallback_temperature: float = Field(
        0.1, validation_alias="SYNTHESIZER_FALLBACK_TEMP"
    )
