import json
import pathlib
from typing import Any, Generator, List, Sequence

import tiktoken
from langchain_core.documents import BaseDocumentTransformer, Document
import wandb

from bot338.utils import get_logger
from bot338.ingestion.preprocessors.bills import BillsTransformer

logger = get_logger(__name__)


class Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


tokenizer = Tokenizer("gpt-4o-2024-08-06")


def length_function(content: str) -> int:
    return len(tokenizer.encode(content))


def len_function_with_doc(document: Document) -> int:
    return len(tokenizer.encode(document.page_content))


class DocumentTransformer(BaseDocumentTransformer):
    def __init__(
        self,
        max_size: int = 5000,
        model_name: str = "gpt-4o-2024-08-06",
        length_function=None,
    ) -> None:
        self.chunk_size = max_size
        self.length_function = length_function
        self.bill_transformer = BillsTransformer(
            chunk_size=self.chunk_size, model_name=model_name
        )

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
        return self.bill_transformer.transform_documents(documents)


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
    max_size: int = 5000
    model_name: str = "gpt-4o-2024-08-06"
    doc_transformer = DocumentTransformer(max_size=max_size, model_name=model_name)

    result_artifact = wandb.Artifact(result_artifact_name, type="dataset")

    for document_file in documents_files:
        # print(f"{document_file.parent.name=}")  # bills
        # print(f"{document_file.name=}")  # documents.jsonl
        # document_file=PosixPath('/Users/tesla/Documents/project/consulting/rag for reporters/repo/338.ai/artifacts/dojo_mode:v0/bills/documents.jsonl')

        with document_file.open("r") as f:
            documents = [Document(**json.loads(line)) for line in f]
            transformed_documents = process_documents_file(documents, doc_transformer)

            config = json.load((document_file.parent / "config.json").open())
            metadata = json.load((document_file.parent / "metadata.json").open())
            cache_dir = (
                pathlib.Path(config["data_source"]["cache_dir"]).parent
                / "transformed_data"
            )

            transformed_file = (
                cache_dir / document_file.parent.name / document_file.name
            )
            logger.info(f"{transformed_file=}")
            # transformed_file=PosixPath('data/cache/transformed_data/bills/documents.jsonl')

            transformed_file.parent.mkdir(parents=True, exist_ok=True)
            with transformed_file.open("w") as of:
                for document in transformed_documents:
                    of.write(json.dumps(document.model_dump()) + "\n")

            config["chunk_size"] = max_size
            with open(transformed_file.parent / "config.json", "w") as of:
                json.dump(config, of)

            metadata["num_transformed_documents"] = len(transformed_documents)
            with open(transformed_file.parent / "metadata.json", "w") as of:
                json.dump(metadata, of)

            result_artifact.add_dir(
                str(transformed_file.parent), name=document_file.parent.name
            )
            logger.info(f"{str(transformed_file.parent)=}")
            # str(transformed_file.parent)='data/cache/transformed_data/bills'
            logger.info(f"{document_file.parent.name=}")
            # document_file.parent.name='bills'

    run.log_artifact(result_artifact)
    run.finish()
    return f"{entity}/{project}/{result_artifact_name}:latest"


if __name__ == "__main__":
    from langchain_text_splitters import CharacterTextSplitter
    from bot338.ingestion.preprocessors.bills import BillsTransformer

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
        btransformer = BillsTransformer(chunk_size=50)

        for chunk in btransformer.transform_documents([sample_doc]):
            print(chunk.page_content)
            print(
                "\n--------------------------------------------------------------------------------->>>>"
            )

        print(chunk.metadata)
