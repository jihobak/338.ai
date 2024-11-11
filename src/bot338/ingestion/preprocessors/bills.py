from typing import Any, Callable, List, Optional, Sequence, Tuple
import uuid
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever


class BillTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        chunk_size: int = 8000,
        chunk_overlap: int = 0,
        model_name: str = "gpt-4o-2024-08-06",
    ):
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
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


class BillTransformerForParentRetriever(BaseDocumentTransformer):
    """

    - 현재 문서 변환 단계에서 고려하는 대상 모델은 openai 으로 제한한다.
    - 기본적으로 처리 로직은 ParentDocumentRetriever 에서 쪼개서 저장하는 방식을 따른다.
      - 여기서 parent splitter 사용은 고려하지 않는다.
    """

    def __init__(
        self,
        chunk_size: int = 8000,
        chunk_overlap: int = 0,
        model_name: str = "gpt-4o-2024-08-06",
        parent_id_key: str = "parent_id",
        child_metadata_fields: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            parent_id_key: Parent document 의 id
            child_metadata_fields: Metadata fields to leave in child documents. If None, leave all parent document metadata.

        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap = chunk_overlap

        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.parent_id_key = parent_id_key
        self.child_metadata_fields = child_metadata_fields
        self.child_splitter = self.recursive_splitter
        self.parent_splitter = None

    def _split_docs_for_adding(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        """이 코드는 langchain 의 ParentDocumentRetriever 의 코드이다."""
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                """
                ID를 생성할 때는 그 목적이 문서를 저장하고 관리하는 데 있습니다.
                add_to_docstore가 False인 경우 문서를 docstore에 추가하지 않는다는 의미인데, 그렇다면 ID가 필요하지 않습니다.
                """
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                # parent document 의 id 를 추가한다.
                _doc.metadata[self.parent_id_key] = _id

                # sub doc 의 id 를 추가한다.
                """
                TODO
                    - sub_doc 에 'id' 가 필요엾을 수 있다.
                """
                _doc.metadata["id"] = str(uuid.uuid4())

            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        return docs, full_docs

    def transform_documents(
        self,
        documents: Sequence[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A sequence of transformed Documents.
        """

        # split_documents: Sequence[Document] = (
        #     self.recursive_splitter.transform_documents(documents)
        # )

        docs, full_docs = self._split_docs_for_adding(documents, ids, add_to_docstore)

        return docs, full_docs
