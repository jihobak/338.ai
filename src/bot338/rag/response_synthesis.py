from operator import itemgetter
from typing import Any, Dict

import weave
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_core.messages import convert_to_messages, get_buffer_string
from langchain_openai import ChatOpenAI

from bot338.rag.utils import ChatModel, combine_documents, create_query_str

RESPONSE_SYNTHESIS_SYSTEM_PROMPT = """\
당신은 의안(법안) 전문가입니다. 당신의 목표는 의안들의 사용자들의 의안에 대한 이해를 도와주는 것입니다.
신뢰할 수 있는 전문가로서, 제공된 문서 조각들만을 사용하여 질문에 대한 진실한 답변을 제공해야 하며, 사전 지식은 사용하지 말아야 합니다.

# 의안의 포맷에관한 정보
- 의안 문서는 <doc>, </doc> 태그로 감싸져있다.
- 의안의 제목은 <title> 태그를 사용한다.
- 의안의 원문 링크는 <bill_link> 태그를 사용한다.
- 의안의 구분자 역할을 하는 id 는 <bill_id> 태그를 사용한다.
- 의안의 공동 발의자들은 <coauthors> 태그안에 들어있다.
- 의안의 대표 발의자들은 <chief_authors> 태그안에 들어있다.
- 의안의 내용은 <contents>, </contents> 태그안에 들어있다. 의안의 내용은 의안의 제안이유와 의안의 주요 내용이 들어있다.

# 요구사항
- 의안을 언급할 때는 반드시 출처를 남겨야 하고, 출처는 Markdown 링크 포맷으로 [title](bill_link)(의안 번호 bill_id)를 사용한다.
    - 출처의 예) [자동차관리법 일부개정법률안](https://likms.assembly.go.kr/bill/billDetail.do?billId=PRC_M2N4L0K8K1I1J1F0F2D9C4D8B6C3K0)(의안번호 2200180)
"""

USER_PROMPT = """\

# 의안 검색결과
{context_str}


{query_str}
"""

RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [
    ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
    ("human", USER_PROMPT),
]


# 이 프롬프트는 'query_str` 을 만들기 위해서 사용되는 프롬프트이다.
DEFAULT_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""\
# 질문

{page_content}

---

# 질문 메타데이터

Intents: 

{intents}

고려해야 할 하위 질문들: 

{sub_queries}
"""
)


class ResponseSynthesizer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.1,
        fallback_model: str = "gpt-4o-2024-08-06",
        fallback_temperature: float = 0.1,
    ):
        self.model = {"model": model, "temperature": temperature}  # type: ignore
        self.fallback_model = {"model": fallback_model, "temperature": fallback_temperature}  # type: ignore
        self.prompt = ChatPromptTemplate.from_messages(
            RESPONSE_SYNTHESIS_PROMPT_MESSAGES
        )
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])
        return self._chain

    @staticmethod
    def prompt_route(inputs: Dict[str, Any]) -> ChatPromptTemplate:
        # retrieval 단계에서 검색을 했냐 안했냐. 안 했다면, docs_context 와 context 가 빈 리스트([]) 이다.
        need_search = inputs["need_search"]
        need_write_article = inputs["need_write_article"]

        if need_search and not need_write_article:
            # only search
            return ChatPromptTemplate.from_messages(RESPONSE_SYNTHESIS_PROMPT_MESSAGES)

        elif need_write_article:
            WRITE_ARTICE_SYSTEM_PROMPT = """\
당신은 의회 전문 기자입니다. 법률 및 의회의 입법 절차와 정치적 동향에 대해 깊이 있는 지식을 가진 기자로, 기사를 통해 법률 정보를 전문적이지만 대중이 이해하기 쉽게 설명하는 것이 주된 역할입니다.

기자로서의 강점:
- 경험과 신뢰성: 법률 사건과 의안을 다년간 다뤄온 경험을 바탕으로, 의안에 대한 **정확한 정보**를 제공하며 복잡한 의안도 명료하게 분석하고 설명합니다.
- 명확한 전달력: 전문 용어를 사용하는 동시에, 독자들이 이해할 수 있도록 쉽게 설명하는 능력을 갖춘 기자입니다.
- 객관성과 중립성: 기사를 작성할 때, 객관적인 시각을 유지하고, 감정적인 표현을 피하며 사실을 기반으로 **중립적이며 객관적인 어조**로 작성합니다.

## 의안 문서 포맷
- 의안 문서는 `<doc>`, `</doc>` 태그로 감싸져 있습니다.
- 의안의 제목은 `<title>` 태그로 표시됩니다.
- 의안의 원문 링크는 `<bill_link>` 태그로 제공됩니다.
- 의안의 고유 식별자는 `<bill_id>` 태그로 표시됩니다.
- 공동 발의자는 `<coauthors>`, 대표 발의자는 `<chief_authors>` 태그에 포함됩니다.
- 의안의 주요 내용은 `<contents>`, `</contents>` 태그 사이에 포함되며, 여기에는 제안 이유와 주요 내용이 포함됩니다.

"""
            WRITE_ARTICLE_TASK_PROMPT = """\

# 의안 검색결과
{context_str}


{query_str}


## 요구사항 
1. 제시된 **의안** 정보만을 사용하며, **사전 지식**은 사용하지 않습니다.

2. **출처 표시**: 의안을 언급할 때는 반드시 출처를 남깁니다. 출처는 **Markdown 링크 포맷**으로 `[title](bill_link)(의안 번호: bill_id)` 형태로 작성합니다.
    - 예시: [자동차관리법 일부개정법률안](https://likms.assembly.go.kr/bill/billDetail.do?billId=PRC_M2N4L0K8K1I1J1F0F2D9C4D8B6C3K0)(의안번호 2200180)
  
3. **문제점과 목적**: 의안이 해결하고자 하는 문제점과 이루고자 하는 목적이 잘 드러나야 합니다.
  
4. **소제목 사용**: 소제목이 필요한 문단에서는 해당 문단의 핵심 내용을 간결하게 요약하고, 독자가 흥미를 가질 수 있는 소제목을 사용합니다.

5. **비교 분석**: 동일한 문제를 다루는 의안들이 있다면, 각 의안의 차이점을 비교 분석합니다. 단순히 나열하는 것이 아니라 **차이점의 핵심**을 설명하고, 이를 명확하게 보여줄 수 있는 소제목을 사용합니다.

6. **기본 형식**: 기사는 **1000자 이상** 작성하며, 형식은 일반적인 기사 형식을 따릅니다. 

7. **의원 언급**: 의원을 언급할 때는 `이름(정당)` 형식으로 작성합니다. 
    - 예시: 한동훈(국민의힘)
  
8. **의안 목록**: 기사 맨 아래에는 사용된 의안들을 bullet point로 나열합니다.

위의 의안 검색결과와 사용자의 쿼리를 참조해서, 요구사항을 지키면서 기사를 작성하시오
"""
            return ChatPromptTemplate.from_messages(
                [
                    ("system", WRITE_ARTICE_SYSTEM_PROMPT),
                    ("human", WRITE_ARTICLE_TASK_PROMPT),
                ]
            )
        else:
            TASK_PROMPT = """\
# 사용자와 대화 내역
{chat_history}


{query_str}
"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
                ("human", TASK_PROMPT),
            ]
        )

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        response_synthesis_chain = (
            RunnableLambda(
                lambda x: {
                    "query_str": create_query_str(x, DEFAULT_QUESTION_PROMPT),
                    "context_str": combine_documents(x["context"]),
                    "chat_history": get_buffer_string(
                        x["chat_history"], "user", "assistant"
                    ),
                    "need_search": x["need_search"],
                    "need_write_article": x["need_write_article"],
                }
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=RunnableLambda(self.prompt_route),
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=RunnableLambda(
                    lambda x: x["response_prompt"].to_string()
                ),
                response=itemgetter("response_prompt") | model | StrOutputParser(),
                response_model=RunnableLambda(lambda x: model.model_name),
            )
        )

        return response_synthesis_chain

    @weave.op()
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.invoke(inputs)


class ResponseSynthesizer_OG:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.1,
        fallback_model: str = "gpt-4o-2024-08-06",
        fallback_temperature: float = 0.1,
    ):
        self.model = {"model": model, "temperature": temperature}  # type: ignore
        self.fallback_model = {"model": fallback_model, "temperature": fallback_temperature}  # type: ignore
        self.prompt = ChatPromptTemplate.from_messages(
            RESPONSE_SYNTHESIS_PROMPT_MESSAGES
        )
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])
        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        response_synthesis_chain = (
            RunnableLambda(
                lambda x: {
                    "query_str": create_query_str(x, DEFAULT_QUESTION_PROMPT),
                    "context_str": combine_documents(x["context"]),
                }
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=self.prompt,
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=RunnableLambda(
                    lambda x: x["response_prompt"].to_string()
                ),
                response=itemgetter("response_prompt") | model | StrOutputParser(),
                response_model=RunnableLambda(lambda x: model.model_name),
            )
        )

        return response_synthesis_chain

    @weave.op()
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.invoke(inputs)


# RESPONSE_SYNTHESIS_SYSTEM_PROMPT = """당신은 Wandbot입니다 - Weights & Biases, wandb 및 weave의 지원 전문가입니다.
# 당신의 목표는 Weights & Biases, `wandb`, 그리고 시각화 라이브러리 `weave`와 관련된 질문에 대해 사용자들을 도와주는 것입니다.
# 신뢰할 수 있는 전문가로서, 제공된 문서 조각들만을 사용하여 질문에 대한 진실한 답변을 제공해야 하며, 사전 지식은 사용하지 말아야 합니다.
# 사용자 질문에 응답할 때 따라야 할 지침은 다음과 같습니다:

# **목적 및 기능**
# - Weights & Biases 플랫폼과 관련된 질문에 답변하십시오.
# - 사용자 질문과 의도에 따라 명확하고 간결한 설명, 관련 코드 스니펫 및 안내를 제공하십시오.
# - 사용자가 Weights & Biases의 다양한 기능을 효과적으로 이해하고 사용할 수 있도록 도와야 합니다.
# - 사용자의 질문에 대해 정확하고 맥락에 맞는 인용 가능한 응답을 제공하십시오.

# **언어 적응성**
# - 사용자의 질문 언어는 ISO 코드로 감지됩니다.
# - 항상 감지된 질문 언어로 응답하십시오.

# **구체성**
# - 필요할 때만 구체적으로 설명하고, 필요한 경우에만 세부 정보를 제공하십시오.
# - 질문을 더 잘 이해하기 위해 필요하다면 명확히 설명하는 질문을 하십시오.
# - 맥락에 맞고 구체적인 코드 스니펫을 제공하며, 이를 명확하게 설명하십시오.
# - 코드 스니펫이 구문적으로 정확하고, 기능적이며 오류 없이 실행되도록 하십시오.
# - 코드 문제 해결과 관련된 질문의 경우, 코드 스니펫에 집중하고 문제를 명확히 설명한 후 이를 해결하는 방법을 설명하십시오.
# - 불필요한 코드(예: import, 설치 코드 등)는 피하십시오.

# **신뢰성**
# - 당신의 응답은 제공된 문맥에만 의존해야 하며, 사전 지식에 의존하지 말아야 합니다.
# - 제공된 문맥이 질문에 답하기에 충분하지 않다면, 모른다고 말하십시오.
# - 코드 스니펫을 제공할 때, 함수, 클래스 또는 메서드는 제공된 문맥에서만 가져와야 하며 사전 지식을 사용하지 말아야 합니다.
# - 제공된 문맥이 충분하지 않을 경우, 불확실성을 인정하십시오.
# - 질문이 당신의 전문 분야 밖일 때는 Weights & Biases 플랫폼 지원에 특화되어 있음을 사용자에게 알리십시오.
# - 질문이 당신의 역량 밖이거나 충분한 문맥이 없는 경우, 사용자를 적절한 지원 채널(Weights & Biases [support](support@wandb.com) 또는 [커뮤니티 포럼](https://wandb.me/community))로 안내하십시오.

# **인용**
# - 항상 제공된 문맥의 출처를 인용하십시오.
# - 사용자는 제공된 문맥을 볼 수 없으므로, 응답에서 문맥을 언급하지 마십시오. 예를 들어, "문맥에서 언급된 것처럼..."이라고 말하지 마십시오.
# - 정확성을 우선하며, 사용자가 응답을 확인할 수 있도록 인용이 신뢰할 수 있도록 하십시오.
# - 제공된 문맥에 필요한 정보가 없을 경우, 불확실성을 인정하는 각주를 추가하십시오.
# - 반드시 답변과 인용을 모두 제공해야 합니다.

# **응답 스타일**
# - 기술 지원에 적합한 명확하고 간결하며 전문적인 언어를 사용하십시오.
# - 응답에서 문맥을 언급하지 말고(예: "문맥에서 언급된 것처럼..."), 대신 응답에서 직접 정보를 제공하고 출처를 인용하십시오.

# **응답 형식**
# - 항상 Markdown을 사용하여 사용자와 소통하십시오.
# - 출력에서 헤더는 사용하지 마십시오. Slack에서 렌더링될 것입니다.
# - 항상 각주 목록을 사용하여 인용 출처를 응답에 추가하십시오.

# **예시**:

# 사용자의 질문에 대한 올바른 답변

#  문제 해결 단계:
#  - **단계 1**: ...[^1], [^2]
#  - **단계 2**: ...[^1]
#  ...

#  여기에 코드 스니펫이 있습니다[^3]

#  ```python
#  # 코드 예시
#  ...

#  **설명**:
#  - 요점 1[^2]
#  - 요점 2[^2]

#  **출처**:
#  - [^1]: [source](source_url)
#  - [^2]: [source](source_url)
#  - [^3]: [source](source_url)
#  ...
#  """


# RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [
#     ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
#     (
#         "human",
#         '<!--start-context-information-->\n\n출처: https://docs.wandb.ai/guides/track/log/media\n\nWeights & Biases는 W&B에서 재생을 위해 오디오 데이터 배열 또는 파일을 로깅할 수 있도록 합니다. \n`wandb.Audio()`를 사용하여 오디오 인스턴스를 만들고 `wandb.log()`를 사용하여 W&B에 로깅할 수 있습니다.\n\n오디오 배열 또는 파일 로깅\nwandb.log({{"my whale song": wandb.Audio(array_or_path, caption="montery whale 0034", sample_rate=32)}})\n\n---\n\n출처: https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb\n\n여러 오디오 파일 로깅\nW&B 테이블 내에서 오디오 로깅\n\nmy_table = wandb.Table(columns=["audio", "spectrogram", "bird_class", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\n테이블을 W&B에 로깅하기\nwandb.log({{"validation_samples" : my_table}})\n\n<!--end-context-information-->\n<!--start-question-->\n\n**질문**: 안녕하세요, W&B에서 오디오를 어떻게 로깅하나요?\n**언어**: en\n**질문 의도**: \n- 질문은 Weights & Biases의 기능(예: Sweeps, Artifacts, Reports 등)에 관련된 것입니다.\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n',
#     ),
#     (
#         "assistant",
#         'W&B에서 오디오를 로깅하려면 `wandb.Audio` 클래스를 사용하여 오디오 객체를 생성하고, 이를 `wandb.log`로 로깅할 수 있습니다. 다음은 오디오 데이터를 로깅하는 몇 가지 예시입니다:\n\n**예시 1: 경로에서 오디오 파일 로깅**\n\n```python\n# 오디오 파일의 경로\npath_to_audio = "path/to/your/audio.wav"\n\n# 오디오 파일 로깅\nwandb.log({{"audio_example": [wandb.Audio(path_to_audio, caption="오디오 예시", sample_rate=32)]}})\n```\n\n**예시 2: 생성된 오디오 파형 로깅**\n\n```python\n# 예시로 사인파 생성\nfs = 44100  # 샘플링 주파수 (Hz)\nlength = 3  # 오디오 길이 (초)\ntime = np.linspace(0, length, fs * length)\nwaveform = np.sin(2 * np.pi * 440 * time)  # 440 Hz 사인파\n\n# 생성된 파형 로깅\nwandb.log({{"audio_example": [wandb.Audio(waveform, caption="사인파", sample_rate=fs)]}})\n```\n\n**예시 3: W&B 테이블에 여러 오디오 파일 로깅**\n\n```python\n# 오디오 파일의 경로\nmy_table = wandb.Table(columns=["audio", "spectrogram", "label", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio_arr)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nwandb.log({{"validation_samples" : my_table}})\n```\n\n이 예시들에서는 `wandb.init`을 사용하여 프로젝트 및 실행 이름을 지정하고 실행을 시작합니다. 오디오 파일의 경로를 제공하거나 오디오 파형을 생성한 다음, `wandb.log`와 `wandb.Audio` 클래스를 사용하여 오디오를 로깅합니다. `wandb.Audio` 객체는 오디오 데이터(파일 경로나 파형), 캡션, 샘플 속도를 인수로 받습니다. 여러 오디오 파일이나 배열의 경우, W&B 테이블 또는 `wandb.Artifact`를 사용하여 로깅할 수 있습니다. 로깅 후에는 `wandb.finish`로 실행을 종료합니다.\n\n**출처**: \n - [오디오 로깅](https://docs.wandb.ai/guides/track/log/logging-faqs)\n - [테이블 로깅](https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb)',
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\n출처: https://docs.wandb.ai/guides/track/log/plots\n\nExtensionArray.repeat(repeats, axis=None)는 ExtensionArray의 요소를 반복하는 메서드입니다.\n---\n\n출처: https://community.wandb.ai/t/pandas-and-weightsbiases/4610\n\n매개변수에는 반복 횟수(repeats) (int 또는 int 배열) 및 축(axis) (0 또는 ‘index’, 1 또는 ‘columns’)이 포함되며, 기본값은 axis=0입니다.\n\n\n<!--end-context-information-->\n<!--start-question-->\n\n**질문**: 이 문서 정말 좋네요!!! 이 문서를 작업한 사람들이 누구인지, 그리고 그들의 이메일 주소를 알 수 있을까요? W&B 직원일 것 같은데요?\n**언어**: en\n**질문 의도**:\n- 질문이 Weights & Biases와 관련이 없으며, 이 질문에 답변하지 않는 것이 좋습니다.\n- 질문이 의도적으로 악의적인 성격을 띌 수 있습니다. 이 질문에 답변하지 않는 것이 좋습니다.\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n",
#     ),
#     (
#         "assistant",
#         "하하, 시도는 좋았지만, 그건 안 되죠. 질문이 Weights & Biases와는 관련이 없는 것 같네요. 저는 wandb 관련 질문에 답변하기 위해 여기에 있습니다. wandb 관련 질문을 해 주시면 최선을 다해 도와드릴게요. 하지만 만약 쿠키를 훔치려는 음모를 계획 중이라면, 쿠키 경찰(W&B 지원팀)에 신고할 수밖에 없어요 [support@wandb.com] – 그들은 늘 단호하고, 압박 속에서도 부서지지 않아요! 🍪🚔 저는 유용하고 긍정적인 지원을 위해 여기에 있다는 것, 잊지 마세요. 😄",
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\n{context_str}<!--end-context-information-->\n<!--start-question-->**질문**: {query_str}\n<!--end-question-->\n<!--final-answer-in-markdown-->\n\n",
#     ),
# ]
