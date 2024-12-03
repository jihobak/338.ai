from operator import itemgetter
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Sequence, Tuple
import warnings

from langchain_community.storage import MongoDBStore
from langchain_community.vectorstores import LanceDB
from langchain_community.vectorstores.lancedb import to_lance_filter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.stores import BaseStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain.retrievers.multi_vector import SearchType
from langchain_core.runnables.config import run_in_executor
from langchain_text_splitters import TextSplitter
from langchain.storage._lc_store import create_kv_docstore
from pydantic import model_validator

# if TYPE_CHECKING:
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document

import wandb
import weave

from bot338.ingestion.config import VectorStoreConfig
from bot338.retriever.reranking import CohereRerankChain
from bot338.retriever.utils import EmbeddingsModel
from bot338.utils import get_logger


logger = get_logger(__name__)


class CustomVectorStoreRetriever(VectorStoreRetriever):
    """
    TODO
        - 나중에 시도 해 볼 것.
        - as_retriever 메서드로 정적으로 인자 전달 하지 않고 invoke 동적으로 인자 전달을 하기 위한 목적으로 만든 것
        아직 작성된 것이 아니다.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def direct_query(self) -> List[Document]:
        vector_db: VectorStore = self.vectorstore  # LanceDB

        search_kwargs = self.search_kwargs

        results = vector_db._query(
            query=None, **{**search_kwargs, "k": search_kwargs.get("fetch_k")}
        )

        candidates = vector_db.results_to_docs(results)

        return candidates


class HierarchicalDocumentRetriever(VectorStoreRetriever):
    """langchain VectorStoreRetriever 에 다가 ParentDocumentRetriever 코드를 더했다."""

    id_key: str = "parent_id"

    docstore: BaseStore[str, Document]
    """The storage interface for the parent documents"""

    @model_validator(mode="before")
    @classmethod
    def shim_docstore(cls, values: Dict) -> Any:
        byte_store = values.get("byte_store")
        docstore = values.get("docstore")
        if byte_store is not None:
            docstore = create_kv_docstore(byte_store)
        elif docstore is None:
            raise Exception("You must pass a `byte_store` parameter.")
        values["docstore"] = docstore
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """langchain 의 MultiVectorRetriever 의 코드

        Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])

        docs = self.docstore.mget(ids)

        return [d for d in docs if d is not None]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """langchain 의 MultiVectorRetriever 의 코드, Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        return [d for d in docs if d is not None]

    def direct_query(
        self, k: Optional[int] = None, sort: bool = True
    ) -> List[Document]:
        vector_db: VectorStore = self.vectorstore  # LanceDB

        search_kwargs = self.search_kwargs

        results = vector_db._query(
            query=None,
            **{**search_kwargs, "k": k if k else search_kwargs.get("fetch_k")},
        )

        sub_docs = vector_db.results_to_docs(results)

        if sort:
            sort_by_proposal = lambda docs: sorted(
                docs, key=lambda doc: doc.metadata["proposal_date"], reverse=True
            )
            sub_docs = sort_by_proposal(sub_docs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]

    async def adirect_query(
        self, k: Optional[int] = None, sort: bool = True
    ) -> List[Document]:
        """
        TODO
            - run_in_executor 를 _query 에만 사용하고
            - mget -> amget 으로 바꾸고
        """
        return await run_in_executor(None, self.direct_query, k, sort)

    def _split_docs_for_adding(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        """langchain 의 ParentDocumentRetriever 의 코드"""
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
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
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        return docs, full_docs

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> None:
        """lanchain 의 ParentDocumentRetriever 의 코드"""
        docs, full_docs = self._split_docs_for_adding(documents, ids, add_to_docstore)
        await self.vectorstore.aadd_documents(docs, **kwargs)
        if add_to_docstore:
            await self.docstore.amset(full_docs)


class CustomLanceDB(LanceDB):
    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Return VectorStoreRetriever initialized from this VectorStore.

        Args:
            **kwargs: Keyword arguments to pass to the search function.
                Can include:
                search_type (Optional[str]): Defines the type of search that
                    the Retriever should perform.
                    Can be "similarity" (default), "mmr", or
                    "similarity_score_threshold".
                search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                    search function. Can include things like:
                        k: Amount of documents to return (Default: 4)
                        score_threshold: Minimum relevance threshold
                            for similarity_score_threshold
                        fetch_k: Amount of documents to pass to MMR algorithm
                            (Default: 20)
                        lambda_mult: Diversity of results returned by MMR;
                            1 for minimum diversity and 0 for maximum. (Default: 0.5)
                        filter: Filter by document metadata

        Returns:
            VectorStoreRetriever: Retriever class for VectorStore.

        Examples:

        .. code-block:: python

            # Retrieve more documents with higher diversity
            # Useful if your dataset has many similar documents
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            )

            # Fetch more documents for the MMR algorithm to consider
            # But only return the top 5
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 50}
            )

            # Only retrieve documents that have a relevance score
            # Above a certain threshold
            docsearch.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            )
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return CustomVectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)

    def _query(
        self,
        query: Any,
        k: Optional[int] = None,
        filter: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if k is None:
            k = self.limit
        tbl = self.get_table(name)
        if isinstance(filter, dict):
            filter = to_lance_filter(filter)

        prefilter = kwargs.get("prefilter", False)
        query_type = kwargs.get("query_type", "vector")

        if metrics := kwargs.get("metrics"):
            lance_query = (
                tbl.search(query=query, vector_column_name=self._vector_key)
                .limit(k)
                .metric(metrics)
                .where(filter, prefilter=prefilter)
            )
        else:
            lance_query = (
                tbl.search(query=query, vector_column_name=self._vector_key)
                .limit(k)
                .where(filter, prefilter=prefilter)
            )
        if query_type == "hybrid" and self._reranker is not None:
            lance_query.rerank(reranker=self._reranker)

        docs = lance_query.to_arrow()
        if len(docs) == 0:
            warnings.warn("No results found for the query.")
        return docs


class LanceDBWithHierarchicalDocumentRetriever(CustomLanceDB):
    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()

        return HierarchicalDocumentRetriever(vectorstore=self, tags=tags, **kwargs)


class VectorStore:
    """Vector Store for RAG Bot Service"""

    embeddings_model: EmbeddingsModel = EmbeddingsModel()

    def __init__(
        self,
        config: VectorStoreConfig,
    ):
        self.config = config
        self.embeddings_model = {
            "embedding_model_name": self.config.embedding_model_name,
        }  # type: ignore

        # 기본 테이블 네임은 table_name: Optional[str] = "vectorstore",
        self.vector_db = CustomLanceDB(
            uri=str(config.persist_dir),
            embedding=self.embeddings_model,
            table_name=self.config.collection_name,
        )

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        if config.persist_dir.exists():
            return cls(config=config)
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(config.artifact_url)
        else:
            artifact = wandb.run.use_artifact(config.artifact_url)

        _ = artifact.download(root=str(config.persist_dir))

        return cls(config=config)

    @weave.op()
    def as_retriever(self, search_type="mmr", search_kwargs=None, **kwargs):
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs, **kwargs
        )

        # LanceDB 를 고치지 않고 하는 방법
        # tags = kwargs.pop("tags", None) or [] + self.vectorstore._get_retriever_tags()
        # return CustomVectorStoreRetriever(
        #     vectorstore=self.vectorstore,
        #     tags=tags,
        #     search_type=search_type,
        #     search_kwargs=search_kwargs,
        #     **kwargs,
        # )


class RAGVectorStore(VectorStore):
    """Vector Store for RAG Bot Service"""

    embeddings_model: EmbeddingsModel = EmbeddingsModel()

    def __init__(
        self,
        config: VectorStoreConfig,
    ):
        self.config = config
        self.embeddings_model = {
            "embedding_model_name": self.config.embedding_model_name,
        }  # type: ignore

        # 기본 테이블 네임은 table_name: Optional[str] = "vectorstore",
        self.vector_db = LanceDBWithHierarchicalDocumentRetriever(
            uri=str(config.persist_dir),
            embedding=self.embeddings_model,
            table_name=self.config.collection_name,
        )

        self.docstore = None
        self.id_key = None

        if hasattr(config, "docstore_uri"):
            self.docstore = MongoDBStore(
                connection_string=config.docstore_uri,
                db_name=config.docstore_db_name,
                collection_name=config.docstore_collection_name,
            )
        if hasattr(config, "id_key"):
            self.id_key = config.id_key

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        if config.persist_dir.exists():
            return cls(config=config)
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(config.artifact_url)
        else:
            artifact = wandb.run.use_artifact(config.artifact_url)

        _ = artifact.download(root=str(config.persist_dir))

        return cls(config=config)

    @weave.op()
    def as_retriever(
        self,
        search_type="mmr",
        search_kwargs=None,
        **kwargs,
    ) -> HierarchicalDocumentRetriever:
        """
        - as_retriever 대상은 HierarchicalDocumentRetriever 이다.
            - docstore, id_key 가 반드시 필요하다.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_db.as_retriever(
            docstore=self.docstore,
            id_key=self.id_key,
            search_type=search_type,
            search_kwargs=search_kwargs,
            **kwargs,
        )


class SimpleRetrievalEngine:
    """
    TODO
        - `HierarchicalDocumentRetriever` 사용으로인한 변경사항을 반영해야 한다.
    """

    cohere_rerank_chain = CohereRerankChain()

    def __init__(self, vector_store: VectorStore, rerank_models: Optional[dict] = None):
        self.vector_store = vector_store
        self.cohere_rerank_chain = rerank_models  # type: ignore
        self.embeddings_model = self.vector_store.embeddings_model
        self.redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.embeddings_model
        ).transform_documents

    @weave.op()
    def __call__(
        self,
        question: str,
        top_k: int = 5,
        search_type="mmr",
        sources: List[str] = None,
        redundant_filter: bool = False,
        reranking: bool = False,
    ) -> List[dict[str, Any]]:
        self.top_k = top_k
        search_kwargs = {"k": top_k, "fetch_k": 20}

        retriever = self.vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        if redundant_filter:
            retrieval_chain = RunnableParallel(
                question=itemgetter("question"),
                context=(itemgetter("question") | retriever | self.redundant_filter),
            )
        else:
            retrieval_chain = RunnableParallel(
                question=itemgetter("question"),
                context=(itemgetter("question") | retriever),
            )
            # retrieval_chain = itemgetter("question") | retriever

        if reranking:
            retrieval_chain = retrieval_chain | self.cohere_rerank_chain
        else:
            retrieval_chain = retrieval_chain | RunnableLambda(lambda x: x["context"])

        results = retrieval_chain.invoke({"question": question, "top_k": top_k})
        outputs = []
        for result in results:
            result_dict = {
                "text": result.page_content,
                "score": result.metadata.get("relevance_score", 0.0),
                # "score": result.metadata["relevance_score"],
            }
            metadata_dict = {  ####
                k: v
                for k, v in result.metadata.items()
                if k not in ["relevance_score", "source_content", "id", "parent_id"]
            }
            result_dict["metadata"] = metadata_dict
            outputs.append(result_dict)

        return outputs
