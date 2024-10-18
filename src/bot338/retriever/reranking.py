import weave
from langchain_cohere import CohereRerank
from langchain_core.runnables import RunnableBranch, RunnableLambda


class CohereRerankChain:
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __set__(self, obj, value):
        self.multilingual_model = value["multilingual_reranker_model"]

    def __get__(self, obj, obj_type=None):
        if getattr(obj, "top_k") is None:
            raise AttributeError("Top k must be set before using rerank chain")

        @weave.op()
        def load_rerank_chain():
            cohere_rerank = CohereRerank(top_n=obj.top_k, model=self.multilingual_model)

            return RunnableLambda(
                lambda x: cohere_rerank.compress_documents(
                    documents=x["context"], query=x["question"]
                )
            )

        cohere_rerank = load_rerank_chain()

        return cohere_rerank
