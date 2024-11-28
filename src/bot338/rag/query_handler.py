from datetime import datetime
from dateutil.relativedelta import relativedelta
import enum
import json
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from zoneinfo import ZoneInfo
from typing import ClassVar

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
from pydantic import BaseModel, ValidationInfo, Field, field_validator

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
    date_str = three_months_ago.strftime("%Y-%m-%d")

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
    Labels.SEARCH_BILL.value: "쿼리가 국회에서 발의된 의안 검색과 관련이 있다. ex) '국회에서 발의된 전기차 안전관련 의안을 찾아줘'",
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
    """의안 검색을 위한 쿼리"""

    query: str = Field(
        ...,
        description="유사한 문서를 검색하기 위한 쿼리. 이는 쿼리에 답하기 위한 문서를 찾는 데 사용됩니다.",
    )


class ContentSearch(BaseModel):
    """Determines if a search based on bill content is necessary and provides reasoning."""

    requires_content_search: bool = Field(
        True,
        description="Indicates whether a search based on specific topics or bill content is necessary. Set to 'False' if the search only requires bill metadata (bill number, proposer, party).\n"
        "- 'True' examples: 'Find bills related to electric vehicle safety', 'Summarize bills proposed by *** about personal information', 'Analyze bills proposed by *** related to sex crimes'.\n"
        "- 'False' examples: 'Summarize bill number 2201234', 'List all bills proposed by ***', 'Please provide a brief explanation of the bill proposed by ***'.\n"
        "Also set to 'False' for queries completely unrelated to bills or not requiring any search.",
    )

    reasoning: Optional[str] = Field(
        None,
        description="The rationale behind the decision made for 'requires_content_search' in the given query.",
    )

    # requires_content_search: bool = Field(
    #     True,
    #     description="특정 주제나 의안의 내용을 기반으로 검색이 필요한지 여부. 의안의 내용, 주제가 아니라 의안의 메타데이터(의안번호, 발의자, 정당)만 사용해서 검색이 필요하면 'False'.\n"
    #     "- 'True' examples: '전기차 안전 관련 의안을 찾아줘', '*** 의원이 발의한 개인정보관련 의안들에 대해서 요약 정리해줘', '*** 의원이 발의한 성범죄관련 의안에 대해서 분석해줘'.\n"
    #     "- 'False' examples: '의안번호 2201234 의안를 요약해줘', '*** 의원이 발의한 의안에 대해서 정리해줘'\n"
    #     "전혀 의안과 관련이 없거나 검색과 관련이 없는 경우에도 'False' 에 해당한다.",
    # )

    # reasoning: Optional[str] = Field(
    #     None,
    #     description="해당 쿼리에서 'requires_content_search' 에대한 판단을 한 이유",
    # )


class QueryAnalysis(BaseModel):
    """사용자 쿼리 분석"""

    content_search: Optional[ContentSearch] = Field(
        default_factory=lambda: ContentSearch(
            requires_content_search=False, reasoning="내용과 관련된 검색 필요 없음"
        ),
        description="A field containing relevant information about decision of search based on bill content.",
    )

    intents: List[Intent] = Field(
        ...,
        description=f"사용자 쿼리와 관련된 하나 이상의 의도 목록. 여기에 가능한 의도가 나열되어 있습니다:\n{json.dumps(INTENT_DESCRIPTIONS)}",
        min_items=1,
        max_items=5,
    )

    vector_search_queries: Optional[List[VectorSearchQuery]] = Field(
        default=None,
        description="""\
A list of transformed queries for bill search from various perspectives and contexts.
Each transformed query should focus on different aspects, viewpoints, or specific themes within the content.

Transformation rules for queries:
1. High-level conceptual perspective:
    - Use broad categories of policy areas (e.g., environmental policy, economic policy)
    - Focus on overarching policy goals (e.g., sustainable development, strengthening social safety nets)

2. Detailed policy perspective:
    - Highlight specific policy tools or measures (e.g., carbon credit trading, implementing universal basic income)
    - Focus on particular groups or domains (e.g., addressing youth unemployment, improving elder welfare)

3. Problem-solving perspective:
    - Address current social issues or challenges (e.g., solving low birthrate problems, reducing fine dust pollution)
    - Emphasize future-oriented policies (e.g., responding to the Fourth Industrial Revolution, adapting to climate change)

4. Legislative and institutional perspective:
    - Consider the need for amending relevant laws or systems (e.g., revising the Labor Standards Act, restructuring the tax system)
    - Introduce new laws or systems (e.g., enacting the Data Three Laws, implementing regulatory sandboxes)

Exclusions:
- Information about the proposing political party, time details, proposers, or bill processing status.

Each transformed query should retain the core essence of the original query while approaching it from diverse angles. 
This broadens the search scope and ensures the retrieval of bills with high relevance.
""",
        #         description="""
        # 의안의 핵심 내용을 다양한 관점과 맥락에서 검색하기 위한 변형 쿼리 목록.
        # 각 변형 쿼리는 서로 다른 관점이나 세부 주제에 초점을 맞춰야 합니다.
        # 쿼리 변형 규칙:
        # 1. 상위 개념 관점:
        #     - 정책 분야의 대분류 사용
        #     - 포괄적인 정책 목표 중심
        # 2. 세부 정책 관점:
        #     - 구체적인 정책 수단이나 방안
        #     - 특정 대상이나 영역에 초점
        # 3. 문제해결 관점:
        #     - 해결하고자 하는 구체적 문제나 이슈
        #     - 기대효과나 목표 상태
        # 제외 대상:
        # - 발의 정당명, 시간 정보, 발의자 정보, 처리 상태
        # 예시:
        # 원본: "올해, 더불어민주당이 발의한 민생 경제와 관련된 의안을 찾아줘"
        # 올바른 변형:
        # 1. 상위 개념: "국민경제 안정화 종합대책"
        # 2. 세부 정책: "소상공인 금융지원 체계화"
        # 3. 문제해결: "물가상승 가계부담 완화방안"
        #     """,
        min_items=1,
        max_items=10,
    )

    keywords: Optional[List[Keyword]] = Field(
        default=None,
        description="""\
A list of keywords (search terms) related to the user's query.
These keywords help refine and focus the search process.""",
        min_items=1,
        max_items=5,
    )

    article_style: Optional[str] = Field(
        default=None,
        description="""\
Specify the preferred style for the article, if applicable.
This field captures any specific requirements or preferences for the tone, format, or approach of the article to ensure alignment with user expectations.
""",
    )

    sub_queries: Optional[List[SubQuery]] = Field(
        default=None,
        description="""\
A list of sub-queries that break down the main query into smaller, more specific parts.
This helps in exploring detailed aspects of the original query.
""",
        min_items=1,
        max_items=5,
    )

    standalone_query: Optional[str] = Field(
        None,
        description="""\
A rewritten query that can independently provide an answer even when a conversation history is present.
If no conversation history exists, the original query should be directly copied into this field.
""",
    )

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
            keywords = (
                [keyword.keyword for keyword in self.keywords] if self.keywords else []
            )
            sub_queries = (
                [sub_query.query for sub_query in self.sub_queries]
                if self.sub_queries
                else []
            )
            vector_search_queries = (
                [
                    vector_search_query.query
                    for vector_search_query in self.vector_search_queries
                ]
                if self.vector_search_queries
                else []
            )

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

        if self.content_search:
            requires_content_search = self.content_search.requires_content_search
        else:
            requires_content_search = False

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
            "need_write_article": self.need_write_article,
            "article_style": self.article_style,
            "requires_content_search": requires_content_search,
        }


class ComparisonOperator(str, enum.Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUALS = ">="
    LESS_EQUALS = "<="
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"


class ValueCondition(BaseModel):
    value: Any
    operator: ComparisonOperator = Field(
        default=ComparisonOperator.EQUALS,
    )


class ArrayCondition(BaseModel):
    include: Optional[List[str]] = Field(
        default_factory=list, description="포함시켜야할 대상"
    )
    operator: Optional[Literal["AND", "OR", "IN", "!=", "="]] = Field(
        default="OR",
        # description="현재 컬럼이 전체 쿼리에서 'OR' 조건인지 'AND' 논리 연산자"
        description="같은 컬럼 내에서 사용되는 비교 연산자 (예: 컬럼이 'name' 일때, name = 'y' AND name='n' OR name='w')",
    )

    @field_validator("include", mode="before")
    @classmethod
    def check_type(cls, v: Union[str, List], info: ValidationInfo):
        if isinstance(v, str):
            return [v]
        elif v is None:
            return []
        else:
            return v

    def to_lancedb_filter(self, field_name: str):
        stmts = []
        params = []
        if self.include:
            for target in self.include:
                array_contain = f"array_contains({field_name}, '%s')"
                stmts.append(array_contain)
                params.append(target)

        in_stmt = f"({f' {self.operator} '.join(stmts)})"

        return in_stmt, params


class SearchCondition(BaseModel):
    """각 컬럼의 검색 조건과, 해당 컬럼 조건이 다른 검색조건과 합쳐질때 필요한 논리 연산자"""

    values: List[Union[str, ValueCondition]]
    operator: Optional[Literal["AND", "OR", "IN", "!=", "="]] = Field(
        default="OR",
        # description="현재 컬럼이 전체 쿼리에서 'OR' 조건인지 'AND' 논리 연산자"
        description="같은 컬럼 내에서 사용되는 비교 연산자 (예: 컬럼이 'name' 일때, name = 'y' AND name='n' OR name='w')",
    )

    def __init__(self, **data):
        # 문자열 값들을 ValueCondition 객체로 변환
        if "values" in data and data["values"]:
            data["values"] = [
                v if isinstance(v, (ValueCondition, dict)) else ValueCondition(value=v)
                for v in data["values"]
            ]
        super().__init__(**data)


# class DateRange(BaseModel):
#     start_date: datetime
#     end_date: datetime = Field(default_factory=lambda: datetime.now())


#     @field_validator("start_date", "end_date")
#     def validate_dates(cls, v):
#         if isinstance(v, str):
#             try:
#                 return datetime.fromisoformat(v)
#             except ValueError:
#                 raise ValueError("날짜는 ISO 형식(YYYY-MM-DD)이어야 합니다")
#         return v
class DateRange(BaseModel):
    SEOUL_TIMEZONE: ClassVar[ZoneInfo] = ZoneInfo("Asia/Seoul")
    DEFAULT_START_DATE: ClassVar[datetime] = datetime(
        2024, 5, 30, tzinfo=SEOUL_TIMEZONE
    )

    start_date: datetime = Field(
        default=DEFAULT_START_DATE,
        description="검색 시작 날짜, 기본값은 2024년 5월 30일 (서울 시간)",
    )
    end_date: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("Asia/Seoul")),
        description="검색 종료 날짜, 기본값은 현재 날짜 (서울 시간)",
    )


class SearchMetaData(BaseModel):
    """Metadata used for filtering and refining bill searches. This metadata helps in constructing more precise and targeted searches for bills.
    - Assembly numbers (e.g., 21st or 22nd National Assembly)
    - Bill IDs
    - Lawmaker names
    - Political party names
    - Date ranges
    - Logical operator for combining search conditions
    """

    assembly_numbers: Optional[SearchCondition] = Field(
        default=None,
        description="Search condition for National Assembly terms, using numeric values. For example, 22 for the 22nd Assembly, 21 for the 21st Assembly.",
    )
    bill_ids: Optional[SearchCondition] = Field(
        default=None, description="Search condition for bill numbers"
    )
    names: Optional[ArrayCondition] = Field(
        default=None,
        description="Search condition for names of National Assembly members",
    )
    parties: Optional[ArrayCondition] = Field(
        default=None, description="Search condition for political party names"
    )
    date_range: DateRange = Field(
        default_factory=lambda: DateRange(), description="Date range for the search"
    )
    conditions_operator: Literal["AND", "OR"] = Field(
        default="AND",
        description="Logical operator for combining search condition groups",
    )

    def _build_comparison_condition(
        self, field: str, condition: ValueCondition
    ) -> tuple[str, list]:
        if isinstance(condition, list):
            placeholders = ", ".join(["%s"] * len(condition))
            sql = f"{field} IN ({placeholders})"
            params = [f"'{cond.value}'" for cond in condition]
        elif condition.operator in [ComparisonOperator.IN, ComparisonOperator.NOT_IN]:
            if not isinstance(condition.value, (list, tuple)):
                raise ValueError(
                    f"{condition.operator} 연산자는 리스트 값이 필요합니다"
                )
            placeholders = ", ".join(["%s"] * len(condition.value))
            sql = f"{field} {condition.operator.value} ({placeholders})"
            params = condition.value
        elif condition.operator == ComparisonOperator.LIKE:
            sql = f"{field} LIKE '%s'"
            params = [f"%{condition.value}%"]
        elif condition.operator == ComparisonOperator.NOT_LIKE:
            sql = f"{field} LIKE '%s'"
            params = [f"%{condition.value}%"]
        else:
            sql = f"{field} {condition.operator.value} %s"
            params = [f"'{condition.value}'"]
        return sql, params

    def to_sql_condition(self) -> tuple[str, list]:
        conditions = []
        params = []

        # 국회 대수 조건 처리
        # if self.assembly_numbers and self.assembly_numbers.values:
        #     assembly_number_conditions = []
        #     for value_condition in self.assembly_numbers.values:
        #         if isinstance(value_condition, str):
        #             value_condition = ValueCondition(value=value_condition)
        #         sql, new_params = self._build_comparison_condition(
        #             "assembly", value_condition
        #         )
        #         assembly_number_conditions.append(sql)
        #         params.extend(new_params)

        #     assembly_group = f"({(' ' + self.assembly_numbers.operator + ' ').join(assembly_number_conditions)})"
        #     conditions.append(assembly_group)

        # 의안 번호 조건 처리
        bill_conditions = []
        if self.bill_ids and self.bill_ids.values:
            if self.bill_ids.operator == "IN":
                sql, new_params = self._build_comparison_condition(
                    "metadata.billcode", self.bill_ids.values
                )
                bill_conditions.append(sql)
                params.extend(new_params)
            else:
                for value_condition in self.bill_ids.values:
                    if isinstance(value_condition, str):
                        value_condition = ValueCondition(value=value_condition)
                    sql, new_params = self._build_comparison_condition(
                        "metadata.billcode", value_condition
                    )
                    bill_conditions.append(sql)
                    params.extend(new_params)
            bill_group = (
                f"({(' ' + self.bill_ids.operator + ' ').join(bill_conditions)})"
            )
            conditions.append(bill_group)

        # 이름 조건 처리
        if self.names:
            sql, new_params = self.names.to_lancedb_filter("metadata.chief_authors")
            params.extend(new_params)

            name_group = sql
            conditions.append(name_group)

        # 정당 조건 처리
        if self.parties:
            sql, new_params = self.parties.to_lancedb_filter("metadata.parties")

            params.extend(new_params)
            name_group = sql
            conditions.append(name_group)

        # 날짜 범위 조건 처리
        if self.date_range and (not bill_conditions):
            date_conditions = [
                "metadata.proposal_date >= date '%s'",
                "metadata.proposal_date <= date '%s'",
            ]
            params.extend(
                [self.date_range.start_date.date(), self.date_range.end_date.date()]
            )
            date_group = f"({' AND '.join(date_conditions)})"
            conditions.append(date_group)

        if not conditions:
            return "1=1", []

        final_condition = f" {f' {self.conditions_operator} '.join(conditions)} "
        return final_condition, params


class EnhancedQuery(BaseModel):
    """향상된 쿼리와 쿼리의 메타 데이터 분석"""

    query_analysis: QueryAnalysis = Field(
        default_factory=lambda: QueryAnalysis(
            intents=[
                Intent(label="other", reasoning="사용자의 의도를 파악할 수 없습니다")
            ]
        )
    )
    search_metadata: Optional[SearchMetaData] = Field(
        default_factory=lambda: SearchMetaData(),
        description="""
        Metadata for refining bill searches, including various filtering criteria
        """,
    )

    def parse_output(
        self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """Parse the output of the model"""
        try:
            query_analysis = self.query_analysis.parse_output(query, chat_history)
            if self.search_metadata:
                stmt, data = self.search_metadata.to_sql_condition()
        except Exception as e:
            logger.error(e)
            default_enhanced_query = EnhancedQuery()
            query_analysis = default_enhanced_query.query_analysis
            stmt, data = default_enhanced_query.search_metadata.to_sql_condition
        finally:
            return {
                **query_analysis,
                **{
                    "search_metadata": (
                        (stmt % tuple(data)).strip() if self.search_metadata else None
                    )
                },
            }


ENHANCER_SYSTEM_PROMPT = """\
당신은 의안(법안) 관련 전문가이며, 사용자로부터의 질문을 개선하는 역할을 맡고 있습니다. 당신에게는 대화 내용과 후속 질문이 주어집니다.
당신의 목표는 사용자 질문을 개선하고 제공된 도구를 사용하여 그것을 표현하는 것입니다. 모든 답변은 반드시 **한국어**로 합니다.


[# 주의 사항]
    - 대화 기록이 존재하고, 사용자의 쿼리가 대화 기록속 검색된 의안 결과를 가지고 기사를 작성을 요구하는 경우 도구(tool) 사용시, 원래 의안 검색에 사용되었던 맥락에 충분히 고려해야한다."
    - 현재 국회는 22대 국회이며, 사용자가 질문하는 시점은 {today} 입니다.
    - 22대 국회의 여당은 '국민의힘' 이고 야당은 '더불어민주당(더불어 민주당)', '개혁신당', '진보당', '기본소득당', '조국혁신당' 입니다. 물론 무소속인 국회의원도 있습니다.
    - '최근' 이라함은, 최근 3개월을 의미한다.
    - 쿼리 분석을 할 때, '22대' 처럼 의회 대수를 직접 언급하지 않는 이상 대수를 검색 조건으로 고려할 필요 없다.
    - 쿼리에 따로 검색 날짜를 파악할 수 없는 경우 시작날짜를 '2024-5-30' 로 설정한다.

[# 대한 민국 국회와 정부를 이해하는데 필요한 배경지식]
    - '특검'은 특별검사(special prosecutor)를 의미한다. 따라서 '특검' 대신에 '특별검사'라는 표현을 사용한다.

    
[# 대화 기록]

{chat_history}

"""


ENHANCER_PROMPT_MESSAGES = [
    ("system", ENHANCER_SYSTEM_PROMPT),
    ("human", "질문: {query}"),
]


class QueryEnhancer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        fallback_model: str = "gpt-4o-2024-08-06",
        fallback_temperature: float = 0.0,
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
            today=RunnableLambda(lambda x: datetime.now(ZoneInfo("Asia/Seoul"))),
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
