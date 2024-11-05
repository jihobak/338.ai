import json
import os
import pathlib
from typing import Any, Generator, List, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_community.storage import MongoDBStore
from langchain.retrievers import ParentDocumentRetriever

import wandb

from bot338.ingestion.config import VectorStoreConfig
from bot338.utils import get_logger, make_document_tokenization_safe
from bot338.ingestion.preprocessors.bills import (
    BillTransformer,
    BillTransformerForParentRetriever,
)

logger = get_logger(__name__)


# class Tokenizer:
#     def __init__(self, model_name):
#         self.tokenizer = tiktoken.encoding_for_model(model_name)

#     def encode(self, text: str):
#         return self.tokenizer.encode(text, allowed_special="all")

#     def decode(self, tokens):
#         return self.tokenizer.decode(tokens)


# tokenizer = Tokenizer("gpt-4o-2024-08-06")


# def length_function(content: str) -> int:
#     return len(tokenizer.encode(content))


# def len_function_with_doc(document: Document) -> int:
#     return len(tokenizer.encode(document.page_content))


class DocumentTransformer(BaseDocumentTransformer):
    """여러 타입의 문서를 최종 전처리하는 클래스"""

    def __init__(
        self,
        max_size: int = 5000,
        overlap_size: int = 0,
        model_name: str = "gpt-4o-2024-08-06",
        parent_id_key: str = "parent_id",
    ) -> None:
        self.chunk_size = max_size
        self.overlap_size = overlap_size
        self.model_name = model_name
        self.parent_id_key = parent_id_key

        self.bill_transformer = BillTransformerForParentRetriever(
            chunk_size=self.chunk_size,
            model_name=model_name,
            chunk_overlap=overlap_size,
            parent_id_key=parent_id_key,
        )
        self.full_docs = []

    def fix_metadata_values(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Generator[Document, None, None]:
        for doc in documents:
            metadata = doc.metadata
            # dict.items()를 리스트로 변환하여 순회와 수정 작업 분리
            for k, v in list(metadata.items()):
                if v is None:
                    metadata[k] = ""  # None 값을 빈 문자열로 변경

            yield doc

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A sequence of transformed Documents.
        """
        # after_fix_metadatas = list(self.fix_metadata_values(documents))
        transformed_documents = []
        for document in list(documents):
            """
            TODO
                - embedding model, llm model 여부에 따른 `make_document_tokenization_safe` 사용고민
            """
            document = make_document_tokenization_safe(
                document, model_name=self.model_name
            )
            if document.metadata.get("source_type", "") == "bill":
                if type(self.bill_transformer) == BillTransformerForParentRetriever:
                    docs, full_docs = self.bill_transformer.transform_documents(
                        [document]
                    )
                    # if len(docs) > 1:
                    #     _, full_doc = full_docs[0]
                    #     print(
                    #         f"{len(docs)=}, {len(full_docs)}, {full_doc.metadata['bill_no']=}"
                    #     )
                    transformed_documents.extend(docs)
                    self.full_docs.extend(full_docs)
                else:
                    transformed_documents.extend(
                        self.bill_transformer.transform_documents([document])
                    )

        return transformed_documents


def process_documents_file(
    documents: List[Document], transformer: DocumentTransformer
) -> List[Document]:
    transformed_documents = transformer.transform_documents(documents)

    return transformed_documents


def load(
    project: str,
    entity: str,
    source_artifact_path: str,
    result_artifact_name: str = "transformed_data",
) -> str:
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=project, entity=entity, job_type="preprocess_data"
    )
    artifact: wandb.Artifact = run.use_artifact(source_artifact_path, type="dataset")
    artifact_dir: str = artifact.download()

    documents_files: List[pathlib.Path] = list(
        pathlib.Path(artifact_dir).rglob("documents.jsonl")
    )

    # settings for document transformer
    max_size: int = int(os.getenv("PREPROCESS_CHUNK_SIZE"))
    overlap_size: int = int(os.getenv("PREPROCESS_CHUNK_OVERLAP_SIZE"))
    tokenizer_model_name: str = os.getenv("PREPROCESS_TOKENIZER_MODEL")

    # doc store for origianl documents
    vector_config = VectorStoreConfig()

    doc_transformer = DocumentTransformer(
        max_size=max_size,
        overlap_size=overlap_size,
        model_name=tokenizer_model_name,
        parent_id_key=vector_config.id_key,
    )

    docstore = MongoDBStore(
        vector_config.docstore_uri,
        db_name=vector_config.docstore_db_name,
        collection_name=vector_config.docstore_collection_name,
    )

    result_artifact = wandb.Artifact(result_artifact_name, type="dataset")

    for document_file in documents_files:

        with document_file.open("r") as f:
            documents = [Document(**json.loads(line)) for line in f]
            transformed_documents = process_documents_file(documents, doc_transformer)

            config = json.load((document_file.parent / "config.json").open())
            metadata = json.load((document_file.parent / "metadata.json").open())
            cache_dir = (
                pathlib.Path(config["data_source"]["cache_dir"]) / result_artifact_name
            )

            transformed_file = (
                cache_dir / document_file.parent.name / document_file.name
            )
            # transformed_file=PosixPath('data/cache/transformed_dev/bill/documents.jsonl')

            transformed_file.parent.mkdir(parents=True, exist_ok=True)
            with transformed_file.open("w") as of:
                for document in transformed_documents:
                    of.write(json.dumps(document.model_dump()) + "\n")

            # full doc 이 존재 할 경우
            if doc_transformer.full_docs:
                # save to docstore
                docstore.mset(doc_transformer.full_docs)

                full_docs_file = (
                    cache_dir / document_file.parent.name / "full_documents.jsonl"
                )
                with full_docs_file.open("w") as of:
                    for _, doc in doc_transformer.full_docs:
                        of.write(json.dumps(doc.model_dump()) + "\n")

            config["chunk_size"] = max_size
            config["overlap_size"] = overlap_size
            config["tokenizer_model_name"] = tokenizer_model_name
            with open(transformed_file.parent / "config.json", "w") as of:
                json.dump(config, of)

            metadata["num_transformed_documents"] = len(transformed_documents)
            with open(transformed_file.parent / "metadata.json", "w") as of:
                json.dump(metadata, of)

            result_artifact.add_dir(
                str(transformed_file.parent), name=document_file.parent.name
            )

    run.log_artifact(result_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


if __name__ == "__main__":
    from langchain_text_splitters import CharacterTextSplitter
    from bot338.ingestion.preprocessors.bills import BillTransformer

    file_path = "/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/artifacts/dojo_mode:v0/bills/documents.jsonl"
    # fastapi pydantic weave openai langchain langchain-openai langchain-chroma wandb pydantic-settings
    # pandas numpy
    # langchain-text-splitters tiktoken
    with open(file_path, "r") as f:
        documents = [Document(**json.loads(line)) for line in f]
        sample_doc = documents[0]
        sample_content = sample_doc.page_content
        print(f"{sample_content}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        btransformer = BillTransformer(chunk_size=50)

        for chunk in btransformer.transform_documents([sample_doc]):
            print(chunk.page_content)
            print(
                "\n--------------------------------------------------------------------------------->>>>"
            )

        print(chunk.metadata)
