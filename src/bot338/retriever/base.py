from operator import itemgetter
from typing import Any, List, Optional, TYPE_CHECKING
import warnings

from langchain_community.vectorstores import LanceDB
from langchain_community.vectorstores.lancedb import to_lance_filter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.vectorstores import VectorStoreRetriever

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
from bot338.retriever.utils import OpenAIEmbeddingsModel
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
        vector_db = self.vectorstore  # LanceDB

        search_kwargs = self.search_kwargs

        results = vector_db._query(
            query=None, **{**search_kwargs, "k": search_kwargs.get("fetch_k")}
        )

        candidates = vector_db.results_to_docs(results)

        return candidates


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

    @weave.op()
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
        print(f"[LanceDB > _query]{filter=}")
        prefilter = kwargs.get("prefilter", False)
        query_type = kwargs.get("query_type", "vector")

        if metrics := kwargs.get("metrics"):
            print(f"[LanceDB > _query] metrics, {filter=}")
            lance_query = (
                tbl.search(query=query, vector_column_name=self._vector_key)
                .limit(k)
                .metric(metrics)
                .where(filter, prefilter=prefilter)
            )
        else:
            print(f"[LanceDB > _query] else, {filter=}")
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


class VectorStore:
    """Vector Store for RAG Bot Service"""

    embeddings_model: OpenAIEmbeddingsModel = OpenAIEmbeddingsModel()

    def __init__(
        self,
        config: VectorStoreConfig,
    ):
        self.config = config
        self.embeddings_model = {
            "embedding_model_name": self.config.embedding_model_name,
        }  # type: ignore

        # 기본 테이블 네임은 table_name: Optional[str] = "vectorstore",
        self.vectorstore = CustomLanceDB(
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
        return self.vectorstore.as_retriever(
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


class SimpleRetrievalEngine:
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
