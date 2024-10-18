import datetime
import logging
import os
import re
from typing import Any

import tiktoken
from langchain_core.documents import Document


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance with the specified name.
    """
    logging.basicConfig(
        # format="%(asctime)s : %(levelname)s : %(message)s",
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    )
    logger = logging.getLogger(name)
    return logger


logger = get_logger(__name__)


class Timer:
    """A simple timer class for measuring elapsed time."""

    def __init__(self) -> None:
        """Initializes the timer."""
        self.start = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.stop = self.start

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Stops the timer."""
        self.stop = datetime.datetime.now().astimezone(datetime.timezone.utc)

    @property
    def elapsed(self) -> float:
        """Calculates the elapsed time in seconds."""
        return (self.stop - self.start).total_seconds()


def make_document_tokenization_safe(document: Document) -> Document:
    """Removes special tokens from the given documents.

    Args:
        documents: A list of strings representing the documents.

    Returns:
        A list of cleaned documents with special tokens removed.
    """

    # 'cl100k_base' is embedding model for 'text-embedding-3-large', 'text-embedding-3-small'
    encoding = tiktoken.get_encoding("cl100k_base")
    special_tokens_set = encoding.special_tokens_set

    def remove_special_tokens(text: str) -> str:
        """Removes special tokens from the given text.

        Args:
            text: A string representing the text.

        Returns:
            The text with special tokens removed.
        """
        for token in special_tokens_set:
            text = text.replace(token, "")
        return text

    content = document.page_content
    cleaned_document = remove_special_tokens(content)
    return Document(page_content=cleaned_document, metadata=document.metadata)


def clean_document_content(doc: Document) -> Document:
    cleaned_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)
    cleaned_content = cleaned_content.strip()
    cleaned_document = Document(page_content=cleaned_content, metadata=doc.metadata)
    cleaned_document = make_document_tokenization_safe(cleaned_document)
    return cleaned_document
