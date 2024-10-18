"""This module contains the configuration settings for wandbot.

The `ChatConfig` class in this module is used to define various settings for wandbot, such as the model name, 
maximum retries, fallback model name, chat temperature, chat prompt, index artifact, embeddings cache, verbosity, 
wandb project and entity, inclusion of sources, and query tokens threshold. These settings are used throughout the 
chatbot's operation to control its behavior.

Typical usage example:

  from wandbot.chat.config import ChatConfig
  config = ChatConfig()
  print(config.chat_model_name)
"""

import os
import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )
    wandb_project: str | None = Field("338lab", env="WANDB_PROJECT")
    wandb_entity: str | None = Field("trendfollowing", env="WANDB_ENTITY")
    index_artifact: str = Field(
        f"{os.environ.get('WANDB_ENTITY', 'trendfollowing')}/{os.environ.get('WANDB_PROJECT', '338lab')}/bot338_index:latest",
        env="WANDB_INDEX_ARTIFACT",
        validation_alias="wandb_index_artifact",
    )
    # Retrieval settings
    top_k: int = 5
    search_type: str = "mmr"
    # Cohere reranker models
    multilingual_reranker_model: str = "rerank-multilingual-v3.0"
    # Response synthesis settings
    response_synthesizer_model: str = "gpt-4o-2024-08-06"
    response_synthesizer_temperature: float = 0.1
    response_synthesizer_fallback_model: str = "gpt-4o-2024-08-06"
    response_synthesizer_fallback_temperature: float = 0.1
