from typing import Any, Callable, Sequence
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BillsTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        chunk_size: int = 8000,
        chunk_overlap: int = 0,
        model_name: str = "gpt-4o-2024-08-06",
        length_function: Callable[[str], int] = None,
    ):
        self.chunk_size: int = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function: Callable[[str], int] = (
            length_function if length_function is not None else len
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A sequence of transformed Documents.
        """

        # 현재는 로직이 하나의 splitter 에 의존한다.
        split_documents = self.recursive_splitter.transform_documents(documents)
        transformed_documents = []
        for document in split_documents:
            transformed_documents.append(document)

        return transformed_documents
