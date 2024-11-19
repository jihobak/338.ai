from datetime import datetime
from dateutil.relativedelta import relativedelta
import enum
import json
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Tuple

import regex as re
import weave
from langchain_core.messages import convert_to_messages, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from bot338.rag.utils import ChatModel
from bot338.utils import get_logger


logger = get_logger(__name__)

BOT_NAME_PATTERN = re.compile(r"<@U[A-Z0-9]+>|@[a-zA-Z0-9]+")


def clean_question(question: str) -> str:
    cleaned_query = BOT_NAME_PATTERN.sub("", question).strip()
    return cleaned_query


def get_date_filter_query_str(month_ago: int = 3, only_date: bool = True) -> str:
    # 오늘 날짜 가져오기
    today = datetime.now()
    
    # 3개월 전 날짜 계산
    three_months_ago = today - relativedelta(months=month_ago)
    
    # 날짜를 YYYY-MM-DD 형식의 문자열로 변환
    date_str = three_months_ago.strftime('%Y-%m-%d')
    
    if only_date:
        return date_str
    
    else:
        # 쿼리 문자열 생성
        query = f">= date '{date_str}'"
        
        return query


class Labels(str, enum.Enum):
    UNRELATED = "unrelated"
    SEARCH_BILL = "search_bill"
    WRITE_ARTICLE = "write_article"
    NEEDS_MORE_INFO = "needs_more_info"
    OPINION_REQUEST = "opinion_request"
    NEFARIOUS_QUERY = "nefarious_query"
    OTHER = "other"


INTENT_DESCRIPTIONS = {
    Labels.SEARCH_BILL.value: "쿼리가 국회에서 발의된 의안 검색과 관련이 있다.",
    Labels.WRITE_ARTICLE.value: "쿼리가 발의안에 대한 기사 작성 요청과 관련이 있다.",
    Labels.UNRELATED.value: "쿼리가 국회의 발의안 또는 발의안 관련 기사 작성과 관련이 없다.",
    Labels.NEEDS_MORE_INFO.value: "질문에 답변하기 전에 사용자로부터 더 많은 정보가 필요합니다.",
    Labels.OPINION_REQUEST.value: "질문이 의견을 묻고 있습니다.",
    Labels.NEFARIOUS_QUERY.value: "질문이 악의적이며 의안 관련 시스템을 악용하려고 시도하고 있습니다.",
    Labels.OTHER.value: "이 쿼리는 국회의 발의안 및 관련 기사 작성과 관련이 있을 수 있지만, 사용자의 의도를 파악할 수 없습니다. 질문을 다시 표현하도록 요청하는 것이 좋습니다.",
}

QUERY_INTENTS = {
    Labels.SEARCH_BILL.value: """쿼리가 국회에서 발의된 법안을 검색하는 것과 관련되어 있습니다. 사용자가 요청한 법안에 대한 세부 정보를 검색하고, 검색된 의안들의 핵심 내용을 설명해서 검색결과를 제공해준다.

    [**주의 사항**]
      - 검색된 의안의 내용이 사용자의 쿼리(질문, 요청)과 관련이 없을 경우, 검색된 의안이 없음을 사용자에게 친절히 알려주고, 자연스럽게 다시 질문해볼 것을 요청합니다.
    """,
    Labels.WRITE_ARTICLE.value: "쿼리가 발의안에 관한 기사 작성 요청과 관련되어 있습니다. 사용자의 요구사항과 의안내용을 고려해서 훌륭한 기사를 작성합니다.",
    Labels.UNRELATED.value: "이 쿼리는 국회의 발의안 또는 관련 기사 작성과 관련이 없습니다. 해당 주제에 대한 답변을 피하고, 사용자가 질문을 수정하도록 요청하십시오.",
    Labels.NEEDS_MORE_INFO.value: "이 쿼리에 답변하려면 추가 정보가 필요합니다. 사용자가 제공한 정보가 불충분하므로, 추가 질문을 통해 더 많은 정보를 요청한 후 답변하십시오.",
    Labels.OPINION_REQUEST.value: "질문이 의견을 묻고 있습니다. 주관적인 의견보다는 객관적인 정보를 제공하고, 더 깊은 의견이 필요한 경우 사용자가 더 명확한 질문을 하도록 유도하십시오.",
    Labels.NEFARIOUS_QUERY.value: "질문이 악의적일 가능성이 있습니다. 해당 질문에 답변을 피하고, 질문의 악의적인 의도를 지적하는 유머러스하고 신중한 응답을 제공하십시오.",
    Labels.OTHER.value: "이 쿼리는 국회의 발의안 또는 관련 기사와 관련될 수 있지만, 사용자의 의도를 명확히 파악하기 어렵습니다. 질문을 다시 표현하도록 요청하고, 답변을 유보하십시오.",
}


class Intent(BaseModel):
    """사용자 쿼리와 관련된 의도. 사용자 쿼리를 이해하는 데 사용됩니다."""

    reasoning: str = Field(
        ...,
        description="해당 쿼리와 의도를 연결하는 이유",
    )
    label: Labels = Field(..., description="사용자 쿼리와 관련된 의도")


class Keyword(BaseModel):
    """사용자 쿼리와 관련된 검색어"""

    keyword: str = Field(
        ...,
        description="사용자 쿼리와 가장 관련된 적절한 의안들을 검색 하기 위한 키워드(검색어). "
        "이 키워드는 쿼리에 답하기 위해 필요한 의안들을 찾기위해서 사용됩니다.",
    )


class SubQuery(BaseModel):
    """쿼리에 답하기 위해 필요한 정보를 수집하는 데 도움이 되는 하위 쿼리(sub query)"""

    query: str = Field(
        ...,
        description="쿼리에 답하기 위해 필요한 하위 쿼리. 이 쿼리는 사용자 질문에 답하기 위한 "
        "단계들을 정의하는 데 사용됩니다.",
    )


class VectorSearchQuery(BaseModel):
    """벡터 검색을 위한 쿼리"""

    query: str = Field(
        ...,
        description="벡터 공간에서 유사한 문서를 검색하기 위한 쿼리. 이는 쿼리에 답하기 위한 문서를 찾는 데 사용됩니다.",
    )


class PartyEnum(str, enum.Enum):
    democratic = "더불어민주당"
    republican = "국민의힘"
    etc1 = "조국혁신당"
    etc2 = "개혁신당"
    etc3 = "기본소득당"
    etc4 = "진보당"


class Party(BaseModel):
    """정치 정당"""

    name: PartyEnum = Field(..., description="정당 이름")


class SearchMetadata(BaseModel):
    """검색할 때 필터링 등에 사용되는 쿼리 메타데이터"""

    bill_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="의안 번호 목록. ex) ['2201071', '2201369', ...]",
    )
    parties: Optional[List[Party]] = Field(
        default_factory=list, description="정당 이름 목록"
    )
    member_names: Optional[List[str]] = Field(
        default_factory=list, description="국회의원 이름 목록"
    )
    start_date: Optional[str] = Field(None, description="검색 조건의 시작 날짜")
    end_date: Optional[str] = Field(None, description="검색 조건의 종료 날짜")


class EnhancedQuery(BaseModel):
    """향상된 쿼리"""

    intents: List[Intent] = Field(
        ...,
        description=f"사용자 쿼리와 관련된 하나 이상의 의도 목록. 여기에 가능한 의도가 나열되어 있습니다:\n{json.dumps(INTENT_DESCRIPTIONS)}",
        min_items=1,
        max_items=5,
    )

    search_metadata: SearchMetadata = Field(
        description="데이터 베이스 검색시, 필터링을 위해서 사용할 메타데이터"
    )

    keywords: List[Keyword] = Field(
        ...,
        description="사용자 쿼리와 관련된 다양한 키워드(검색어) 목록.",
        min_items=1,
        max_items=2,
    )
    sub_queries: List[SubQuery] = Field(
        ...,
        description="쿼리를 더 작은 부분으로 나누는 하위 쿼리 목록",
        min_items=1,
        max_items=3,
    )
    vector_search_queries: List[VectorSearchQuery] = Field(
        ...,
        description="벡터 공간에서 유사한 문서를 검색하기 위한 다양한 쿼리 목록",
        min_items=1,
        max_items=3,
    )

    standalone_query: str = Field(
        ...,
        description="채팅 기록이 있을 때 독립적으로 답변할 수 있는 재작성된 쿼리. 채팅 기록이 없을 경우, 원본 쿼리가 그대로 복사되어야 합니다.",
    )

    @property
    def filtering(self) -> bool:
        return any([value for key, value in self.search_metadata])

    @property
    def need_search(self) -> bool:
        return any(
            [
                intent.label in [Labels.SEARCH_BILL, Labels.WRITE_ARTICLE]
                for intent in self.intents
            ]
        )

    @property
    def need_write_article(self) -> bool:
        return any([intent.label in [Labels.WRITE_ARTICLE] for intent in self.intents])

    @property
    def avoid_query(self) -> bool:
        """A query that should be avoided"""

        return any(
            [
                intent.label
                in [
                    Labels.NEEDS_MORE_INFO,
                    Labels.NEFARIOUS_QUERY,
                    Labels.OPINION_REQUEST,
                    Labels.UNRELATED,
                    Labels.OTHER,
                ]
                for intent in self.intents
            ]
        )

    def parse_output(
        self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """Parse the output of the model"""
        question = clean_question(query)

        if not chat_history:
            standalone_query = question
        else:
            standalone_query = self.standalone_query

        if self.avoid_query:
            keywords = []
            sub_queries = []
            vector_search_queries = []

        else:
            keywords = [keyword.keyword for keyword in self.keywords]
            sub_queries = [sub_query.query for sub_query in self.sub_queries]
            vector_search_queries = [
                vector_search_query.query
                for vector_search_query in self.vector_search_queries
            ]

        intents_descriptions = ""
        for intent in self.intents:
            intents_descriptions += (
                f'{intent.label.value.replace("_", " ").title()}:'
                f"\n\t{intent.reasoning}"
                f"\n\t{QUERY_INTENTS[intent.label.value]}\n\n"
            )
        all_queries = (
            [standalone_query] + keywords + sub_queries + vector_search_queries
        )

        # filtering
        filter_query_str = get_date_filter_query_str(month_ago=6, only_date=False)
        search_metadata = {"proposal_date": filter_query_str}

        if self.filtering:
            search_metadata: dict = self.search_metadata.model_dump()
            # parties 필드를 value 값으로 변환
            search_metadata["parties"] = [
                party["name"].value for party in search_metadata["parties"]
            ]
            search_metadata["proposal_date"] = filter_query_str

        return {
            "query": query,
            "question": question,
            "standalone_query": standalone_query,
            "intents": intents_descriptions,
            "keywords": keywords,
            "sub_queries": sub_queries,
            "vector_search_queries": vector_search_queries,
            "avoid_query": self.avoid_query,
            "chat_history": chat_history,
            "all_queries": all_queries,
            "need_search": self.need_search,
            "search_metadata": search_metadata,
            "need_write_article": self.need_write_article,
        }


# ENHANCER_SYSTEM_PROMPT = (
#     "당신은 의안(법안) 관련 전문가이며, 사용자로부터의 질문을 개선하는 역할을 맡고 있습니다."
#     "당신에게는 대화 내용과 후속 질문이 주어집니다. "
#     "당신의 목표는 사용자 질문을 개선하고 제공된 도구를 사용하여 그것을 표현하는 것입니다.\n\n"
#     "# 주의 사항\n"
#     "    - 대화 기록이 존재하고, 사용자의 쿼리가 대화 기록속 검색된 의안 결과를 가지고 기사를 작성을 요구하는 경우 도구(tool) 사용시, 원래 의안 검색에 사용되었던 맥락에 충분히 고려해야한다."
#     "\n\n대화 기록: \n\n"
#     "{chat_history}"
# )

ENHANCER_SYSTEM_PROMPT = """\
당신은 의안(법안) 관련 전문가이며, 사용자로부터의 질문을 개선하는 역할을 맡고 있습니다. 당신에게는 대화 내용과 후속 질문이 주어집니다.
당신의 목표는 사용자 질문을 개선하고 제공된 도구를 사용하여 그것을 표현하는 것입니다.


# 주의 사항
    - 대화 기록이 존재하고, 사용자의 쿼리가 대화 기록속 검색된 의안 결과를 가지고 기사를 작성을 요구하는 경우 도구(tool) 사용시, 원래 의안 검색에 사용되었던 맥락에 충분히 고려해야한다."

    
# 대화 기록:"


{chat_history}

"""


ENHANCER_PROMPT_MESSAGES = [
    ("system", ENHANCER_SYSTEM_PROMPT),
    ("human", "질문: {query}"),
    ("human", "!!! 팁: 올바른 형식으로 답변을 작성하세요."),
]


class QueryEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.1,
        fallback_model: str = "gpt-4o-2024-08-06",
        fallback_temperature: float = 0.1,
    ):
        self.model = {"model": model, "temperature": temperature}
        self.fallback_model = {
            "model": fallback_model,
            "temperature": fallback_temperature,
        }
        self.prompt = ChatPromptTemplate.from_messages(ENHANCER_PROMPT_MESSAGES)
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatModel) -> Runnable:
        query_enhancer_chain = self.prompt | model.with_structured_output(EnhancedQuery)

        input_chain = RunnableParallel(
            query=RunnablePassthrough(),
            chat_history=(
                RunnableLambda(lambda x: convert_to_messages(x["chat_history"]))
                | RunnableLambda(lambda x: get_buffer_string(x, "user", "assistant"))
            ),
        )

        full_query_enhancer_chain = input_chain | query_enhancer_chain

        intermediate_chain = RunnableParallel(
            query=itemgetter("query"),
            chat_history=itemgetter("chat_history"),
            enhanced_query=full_query_enhancer_chain,
        )

        chain = intermediate_chain | RunnableLambda(
            lambda x: x["enhanced_query"].parse_output(
                x["query"], convert_to_messages(x["chat_history"])
            )
        )

        return chain

    @weave.op()
    def __call__(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.chain.invoke(inputs)

    @weave.op()
    async def ainvoke(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.chain.ainvoke(inputs)
