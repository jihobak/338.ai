import asyncio
import json
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document
from langchain_openai import ChatOpenAI

# from bot338.retriever.web_search import YouSearchResults
from bot338.utils import clean_document_content


class ChatModel:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, obj_type=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        model = ChatOpenAI(
            model=value["model"],
            temperature=value["temperature"],
            max_retries=self.max_retries,
        )
        setattr(obj, self.private_name, model)


DEFAULT_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""[# 질문 및 사용자의 요구사항]

{page_content}

---

[# 질문 메타데이터]

Intents: 

{intents}

[# 사용자에게 응답하기 위해서 답변하기 전 고려해야 할 하위 질문들:] 

{sub_queries}
---
"""
)


def create_query_str(enhanced_query, document_prompt=DEFAULT_QUESTION_PROMPT):
    page_content = enhanced_query["standalone_query"]
    metadata = {
        "intents": enhanced_query["intents"],
        "sub_queries": "\t" + "\n\t".join(enhanced_query["sub_queries"]).strip(),
    }
    doc = Document(page_content=page_content, metadata=metadata)
    doc = clean_document_content(doc)
    doc_string = format_document(doc, document_prompt)
    return doc_string

    # def build_coactors_prompt(coactors: list[dict]):
    #     # <chinese_char>{ca['chinese_char']}</chinese_char>
    #     members = []
    #     for ca in coactors:
    #         member = f"""\
    # <member>
    #     <name>{ca['name']}</name>
    #     <party>{ca['party']}</party>
    # </member>
    # """
    #         members.append(member.strip())
    #     return "\n".join(members)

    # def build_prompt_for_chief_authors(chief_authors: List[str]):
    #     names = []

    #     for member in chief_authors:
    #         names.append(f"<name>{member}</name>")

    #     names = "\n.".join(names)
    #     return names

    # def build_prompt_for_bill(doc: Document):

    metadata = doc.metadata
    summary = doc.page_content

    bill_no = metadata.get("bill_no", "")
    bill_name = metadata.get("bill_name", "")
    coactors: List[Optional[dict[str, str]]] = metadata.get("coactors", [])
    chief_authors: List[str] = metadata.get("chief_authors", [])

    doc = f"""
<bill>
<bill_no>{bill_no}</bill_no>
<title>{bill_name}</title>
<coauthors>
{build_coactors_prompt(coactors)}
</coauthors>
<cheif_authors>
    {build_prompt_for_chief_authors(chief_authors)}
</cheif_authors>
<contents>
    {summary}
</contents>
</bill>
"""
    return doc


# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
#     template="source: {source}\nsource_type: {source_type}\nhas_code: {has_code}\n\n{page_content}"
# )


def build_coactors_prompt(coactors: list[dict]):
    # <chinese_char>{ca['chinese_char']}</chinese_char>
    members = []
    for ca in coactors:
        member = f"""\t<member><name>{ca['name']}</name><party>{ca['party']}</party></member>"""
        members.append(member)
    return "\n".join(members)


def build_prompt_for_chief_authors(chief_authors: List[str], coactors: list[dict]):
    """

    고려한 샘플들
    - 의원이 아닌 위원장이 발의한 의안의 경우 metadata 의 예는 다음과 같다. 이런 의안의 chief_authors 를 넣을 수 있어야 한다.
    {   ...
        'billcode': '2203263',
        'sponsor_type': '위원장',
        'parties': [],
        'chief_authors': ['국토교통위원장'],
        'coauthors': [],
        'substitue': True,
        'source_type': 'BILL',
        'status': '공포',
        'real_bill': True
    }
    """
    members = []

    if coactors:
        for ca in coactors:
            if ca["name"] in chief_authors:
                member = f"""\t<member><name>{ca['name']}</name><party>{ca['party']}</party></member>"""
                members.append(member)
    else:
        if chief_authors:
            members.extend(chief_authors)

    members = "\n.".join(members)
    return members


def build_prompt_for_bill(doc: Document):

    metadata = doc.metadata
    summary = doc.page_content

    bill_no = metadata.get("billcode", "")
    bill_name = metadata.get("bill_name", "")
    bill_uid = metadata.get("unique_id", "")
    proposal_date = metadata.get("proposal_date", "")
    status = metadata.get("status", "")
    assembly = metadata.get("assembly", "")
    coactors: List[Optional[dict[str, str]]] = metadata.get("coauthors", [])
    chief_authors: List[str] = metadata.get("chief_authors", [])

    doc = f"""\
<bill>
<proposal_date>{proposal_date}</proposal_date>
<bill_id>{bill_no}</bill_id>
<bill_link>https://likms.assembly.go.kr/bill/billDetail.do?billId={bill_uid}</bill_link>
<title>{bill_name}</title>
<cheif_authors>
{build_prompt_for_chief_authors(chief_authors, coactors)}
</cheif_authors>
<contents>
    {summary}
</contents>
</bill>
"""
    return doc


def combine_documents(
    docs,
    # document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n\n---\n\n",
):
    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [build_prompt_for_bill(doc) for doc in cleaned_docs]
    return document_separator.join(doc_strings)


async def acombine_documents(
    docs,
    # document_prompt=DEFAULT_DOCUMENT_PROMPT,
    document_separator="\n\n---\n\n",
):
    if asyncio.iscoroutine(docs):
        docs = await docs

    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [build_prompt_for_bill(doc) for doc in cleaned_docs]
    return document_separator.join(doc_strings)


def process_input_for_retrieval(retrieval_input):
    if isinstance(retrieval_input, list):
        retrieval_input = "\n".join(retrieval_input)
    elif isinstance(retrieval_input, dict):
        retrieval_input = json.dumps(retrieval_input)
    elif not isinstance(retrieval_input, str):
        retrieval_input = str(retrieval_input)
    return retrieval_input


# def get_web_contexts(web_results: YouSearchResults):
#     output_documents = []
#     if not web_results:
#         return []
#     return (
#         output_documents
#         + [
#             Document(page_content=document["context"], metadata=document["metadata"])
#             for document in web_results.web_context
#         ]
#         if web_results.web_context
#         else []
#     )
