import logging
from typing import Any, Dict, List, Optional

import weave
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch
from pydantic import BaseModel

from bot338.retriever.base import VectorStore

logger = logging.getLogger(__name__)


@weave.op()
def reciprocal_rank_fusion(results: list[list[Document]], top_k: int, k=60):
    """
    TODO
        - 조금 더 나은 방법 강구.
    """
    text_to_doc = {}
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_content = doc.page_content
            text_to_doc[doc_content] = doc
            if doc_content not in fused_scores:
                fused_scores[doc_content] = 0.0
            fused_scores[doc_content] += 1 / (rank + k)

    ranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    ranked_results = [text_to_doc[text] for text in ranked_results.keys()]
    return ranked_results[:top_k]


class FusionRetrieval:
    """
    TODO
        - "lambda_mult" 변수를 ChatConifg 에 선언해서, 주입 받도록 해야한다.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        search_type: str = "mmr",
        multilingual_reranker_model: str = "rerank-multilingual-v3.0",  # TODO reranker model 로 변경
    ):
        self.vectorstore = vector_store
        self.top_k = top_k
        self.search_type = search_type

        # self.retriever = self.vectorstore.as_retriever(
        #     search_type=self.search_type,
        #     search_kwargs={"k": self.top_k, "lambda_mult": 0.8},
        # )

        self._chain = None
        self._chain_with_reranking = None
        self.multilingual_reranker_model = multilingual_reranker_model

    @weave.op()
    def rerank_results(
        self,
        queries: List[str],
        context: List[Document],
        top_k: int = 5,
    ):
        reranker = CohereRerank(top_n=top_k, model=self.multilingual_reranker_model)

        query = "\n".join(queries)
        ranked_results = reranker.compress_documents(documents=context, query=query)
        return ranked_results

    @weave.op()
    def retriever_batch(self, queries):
        """wrapped for weave tracking"""
        return self.retriever.batch(queries)

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = RunnablePassthrough().assign(
                docs_context=lambda x: self.retriever_batch(x["all_queries"])
            ) | RunnablePassthrough().assign(
                context=lambda x: reciprocal_rank_fusion(x["docs_context"], self.top_k)
            )
        return self._chain

    @property
    def chain_with_reranking(self) -> Runnable:
        if self._chain_with_reranking is None:
            self._chain_with_reranking = (
                RunnablePassthrough().assign(
                    docs_context=lambda x: self.retriever_batch(x["all_queries"])
                )
                | RunnablePassthrough().assign(
                    fused_context=lambda x: reciprocal_rank_fusion(
                        x["docs_context"], self.top_k * 2
                    )
                )
                | RunnablePassthrough().assign(
                    context=lambda x: self.rerank_results(
                        [x["standalone_query"]], x["fused_context"], self.top_k
                    )
                )
            )
        return self._chain_with_reranking

    @weave.op()
    def route_chain(self, inputs: Dict[str, Any], reranking: bool = False) -> Runnable:
        # 검색이 필요하냐 안 하냐
        need_search = inputs["need_search"]

        # 구체적인 의안 언급시 검색
        def check_bill_ids(inputs) -> bool:
            search_metadata = inputs["search_metadata"]
            bill_ids = []
            if search_metadata:
                bill_ids = search_metadata["bill_ids"]

            return len(bill_ids) > 0

        general_chain = RunnablePassthrough().assign(
            docs_context=lambda x: self.retriever_batch(x["all_queries"])
        ) | RunnablePassthrough().assign(
            context=lambda x: reciprocal_rank_fusion(x["docs_context"], self.top_k)
        )
        skip_chain = RunnablePassthrough().assign(
            docs_context=lambda x: self.retriever.direct_query()
        ) | RunnablePassthrough().assign(context=lambda x: x["docs_context"])
        branch = RunnableBranch(
            (check_bill_ids, skip_chain),
            general_chain,
        )

        if need_search:
            if reranking:
                """
                TODO
                    - direct search 의 경우에도, reranking 적용할지 고민 필요.
                """
                chain = (
                    RunnablePassthrough().assign(
                        docs_context=lambda x: self.retriever_batch(x["all_queries"])
                    )
                    | RunnablePassthrough().assign(
                        fused_context=lambda x: reciprocal_rank_fusion(
                            x["docs_context"], self.top_k * 2
                        )
                    )
                    | RunnablePassthrough().assign(
                        context=lambda x: self.rerank_results(
                            [x["standalone_query"]], x["fused_context"], self.top_k
                        )
                    )
                )
            else:
                chain = branch
                # chain = RunnablePassthrough().assign(
                #     docs_context=lambda x: self.retriever_batch(x["all_queries"])
                # ) | RunnablePassthrough().assign(
                #     context=lambda x: reciprocal_rank_fusion(
                #         x["docs_context"], self.top_k
                #     )
                # )
        else:
            chain = RunnablePassthrough().assign(
                docs_context=lambda x: []
            ) | RunnablePassthrough().assign(context=lambda x: [])

        return chain

    @weave.op()
    def __call__(
        self, inputs: Dict[str, Any], reranking: bool = False
    ) -> Dict[str, Any]:

        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                "k": self.top_k,
                "fetch_k": 10,
                "lambda_mult": 0.8,
                "filter": self.to_lance_filter(inputs.get("search_metadata", None)),
                "prefilter": inputs.get("prefilter", True),
            },
        )
        self._chain = self.route_chain(inputs, reranking)

        return self._chain.invoke(inputs)

        # if reranking:
        #     return self.chain_with_reranking.invoke(inputs)
        # else:
        #     return self.chain.invoke(inputs)

    @staticmethod
    def to_lance_filter(filter: Optional[Dict[str, str]]) -> str | None:
        """Converts a dict filter to a LanceDB filter string."""
        if filter is None:
            return None

        try:
            filter_list = []
            for k, v in filter.items():
                if not v:
                    continue

                if k == "bill_ids":
                    k = "metadata.bill_no"
                    # metadata.bill_no IN ('2200180', '2201071', '2201369', '2201386');
                    v = ", ".join([f"'{bid}'" for bid in v])
                    # v = f"'{'|'.join([f'{bid}' for bid in v])}'"
                    # v = "({})".format('|'.join([f"'{bid}'" for bid in v]))
                    # filter_list.append(f"CAST(regexp_match({k}, {v}) AS BOOLEAN)")
                    filter_list.append(f"({k} IN ({v}))")
                elif k == "parties":
                    k = "metadata.parties"
                    condition = []
                    for party in v:
                        condition.append(f"array_contains({k}, '{party}')")
                    condition_str = " AND ".join(condition)
                    condition_str = f"({condition_str})"
                    filter_list.append(condition_str)
                elif k == "member_names":
                    k = "metadata.chief_authors"  # 나중에 coactors 의 이름만 array 로 따로 만들어서 디비에 넣으면 된다.
                    condition = []
                    for coactor in v:
                        condition.append(f"array_contains({k}, '{coactor}')")
                    condition_str = " OR ".join(condition)
                    condition_str = f"({condition_str})"
                    filter_list.append(condition_str)
                else:
                    k = f"metadata.{k}"
                    filter_list.append(f"{k} = '{v}'")
        except Exception as e:
            logger.error(e)
            return None
        else:
            # return " OR ".join([f"{k} = '{v}'" for k, v in filter.items()])
            return " AND ".join(filter_list)
