import os, re, json, operator, requests
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from itertools import cycle
import threading
import time

# 0. 환경 설정

OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY",   os.environ.get("OPENAI_API_KEY", ""))
GUARDIAN_API_KEY = st.secrets.get("GUARDIAN_API_KEY", os.environ.get("GUARDIAN_API_KEY", ""))
NEWSAPI_KEY      = st.secrets.get("NEWSAPI_KEY",      os.environ.get("NEWSAPI_KEY", ""))


llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)
tutor_llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
quiz_llm       = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)

# 1. 도구 정의

@tool
def check_difficulty(expression: str) -> str:
    """표현의 CEFR 난이도를 반환한다."""
    result = llm.invoke(f"'{expression}'의 CEFR 난이도를 A1~C2 중 하나만 답하라. 이유 없이 레벨만.")
    return result.content.strip()

@tool
def find_synonyms(expression: str) -> str:
    """표현의 유의어 3개를 반환한다."""
    result = llm.invoke(f"'{expression}'의 영어 유의어 3개를 콤마로 구분해 나열하라. 설명 없이 단어만.")
    return result.content.strip()

@tool
def search_guardian(expression: str) -> dict:
    """Guardian API에서 표현이 포함된 기사 문장 2개를 가져온다. 없으면 NewsAPI fallback."""
    url = "https://content.guardianapis.com/search"
    params = {
        "q"          : expression,
        "api-key"    : GUARDIAN_API_KEY,
        "show-fields": "bodyText",
        "page-size"  : 3,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("response", {}).get("results", [])
    except Exception:
        results = []

    sentences, sources = [], []
    pattern = re.compile(re.escape(expression), re.IGNORECASE)

    for item in results:
        body  = (item.get("fields") or {}).get("bodyText", "")
        title = item.get("webTitle", "Guardian")
        for sent in re.split(r'(?<=[.!?])\s+', body):
            if pattern.search(sent) and len(sent) > 30:
                sentences.append(sent.strip())
                sources.append(title)
                break
        if len(sentences) >= 2:
            break

    # ✅ Guardian에서 못 찾으면 NewsAPI fallback
    if len(sentences) < 2 and NEWSAPI_KEY:
        try:
            resp2 = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q"        : expression,
                    "language" : "en",
                    "sortBy"   : "relevancy",
                    "pageSize" : 5,
                    "apiKey"   : NEWSAPI_KEY,
                },
                timeout=10,
            )
            resp2.raise_for_status()
            articles = resp2.json().get("articles", [])
            for article in articles:
                body  = (article.get("content") or article.get("description") or "")
                title = article.get("source", {}).get("name", "NewsAPI")
                for sent in re.split(r'(?<=[.!?])\s+', body):
                    if pattern.search(sent) and len(sent) > 30:
                        sentences.append(sent.strip())
                        sources.append(title)
                        break
                if len(sentences) >= 2:
                    break
        except Exception:
            pass

    # 그래도 없으면 빈 문자열
    while len(sentences) < 2:
        sentences.append(f"No article found for '{expression}'.")
        sources.append("Guardian")

    return {"sentences": sentences, "sources": sources}


# 2. State 정의

class State(TypedDict):
    phase           : str
    expression      : str
    article_title   : str
    article_date    : str
    article_url     : str
    article_body_en : str
    article_body_kr : str
    article_analysis: str
    article_examples: str
    writing_input   : str
    writing_feedback: str
    quiz_question   : str
    quiz_choices    : list
    quiz_answer     : str
    quiz_input      : str
    quiz_feedback   : str
    quiz_retry      : bool
    parallel_results: Annotated[list, operator.add]
    session_log     : Annotated[list, operator.add]
    orchestrator_tasks: list   
    current_task      : str


# 3. 병렬 노드

def node_translate(state: State) -> dict:
    data      = json.loads(state["article_body_en"])
    sentences = data["sentences"]
    joined    = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    result    = llm.invoke(f"다음 영어 문장을 자연스러운 한국어로 번역하라:\n{joined}")
    return {"parallel_results": [{"type": "kr", "content": result.content.strip()}]}

def node_analyze(state: State) -> dict:
    expr   = state["expression"]
    data   = json.loads(state["article_body_en"])
    joined = "\n".join(data["sentences"])
    result = researcher_llm.invoke([   # 메시지 리스트로 전달
        SystemMessage(content="You are an expert English researcher. "
                              "Analyze how expressions are used in real news articles. "
                              "Be precise and academic."),
        HumanMessage(content=f"표현 '{expr}'이 아래 문장에서 어떻게 쓰였는지 "
                             f"한국어로 분석하라 (뉘앙스·맥락 포함):\n{joined}")
    ])
    return {"parallel_results": [{"type": "analysis", "content": result.content.strip()}]}


def node_examples(state: State) -> dict:
    expr   = state["expression"]
    result = llm.invoke(
        f"'{expr}'을 사용한 자연스러운 영어 예문 3개를 번호와 함께 만들어라. "
        f"각 예문 아래에 한국어 해석도 달아라."
    )
    return {"parallel_results": [{"type": "examples", "content": result.content.strip()}]}

# 3-1. Orchestrator-Workers 노드

def orchestrator_node(state: State) -> dict:
    """Orchestrator: 표현 분석에 필요한 작업 목록을 동적으로 생성"""
    expr   = state["expression"]
    result = llm.invoke(f"""
'{expr}' 학습을 위해 필요한 작업을 JSON으로 나열하라.
반드시 아래 3가지를 포함하라:
{{"tasks": ["difficulty_check", "synonym_find", "article_search"]}}
JSON만 출력, 설명 없이.
""")
    try:
        raw   = result.content.strip()
        # 코드블록 제거
        raw   = re.sub(r"```json|```", "", raw).strip()
        tasks = json.loads(raw).get("tasks", [])
    except Exception:
        tasks = ["difficulty_check", "synonym_find", "article_search"]
    return {"orchestrator_tasks": tasks, "parallel_results": []}

def worker_node(state: State) -> dict:
    """Worker: Orchestrator가 지시한 단일 작업 처리"""
    task = state.get("current_task", "")
    expr = state["expression"]

    if task == "difficulty_check":
        res = check_difficulty.invoke({"expression": expr})
        st.session_state["difficulty"] = res          # UI에도 반영
        return {"parallel_results": [{"type": "difficulty", "content": res}]}

    elif task == "synonym_find":
        res = find_synonyms.invoke({"expression": expr})
        st.session_state["synonyms"] = res            # UI에도 반영
        return {"parallel_results": [{"type": "synonyms", "content": res}]}

    elif task == "article_search":
        article   = search_guardian.invoke({"expression": expr})
        sentences = article.get("sentences", [])
        sources   = article.get("sources",   [])
        while len(sentences) < 2: sentences.append("")
        while len(sources)   < 2: sources.append("Guardian")
        body_json = json.dumps(
            {"sentences": sentences, "sources": sources}, ensure_ascii=False
        )
        return {"parallel_results": [{"type": "article_body_en", "content": body_json,
                                      "title": " / ".join(s for s in sources if s)}]}
    return {"parallel_results": []}

def route_workers(state: State):
    """Orchestrator 결과를 Worker들에게 분배"""
    tasks = state.get("orchestrator_tasks", [])
    return [Send("worker_node", {**state, "current_task": t}) for t in tasks]

def node_workers_merge(state: State) -> dict:
    """Worker 결과 통합"""
    results = state.get("parallel_results", [])
    merged  = {}
    for r in results:
        merged[r["type"]] = r.get("content", "")

    # article_body_en 결과 → state에 반영
    article_body = merged.get("article_body_en", "")
    title        = ""
    for r in results:
        if r.get("type") == "article_body_en":
            title = r.get("title", "Guardian")

    return {
        "phase"           : "reading",
        "article_body_en" : article_body,
        "article_title"   : title,
        "parallel_results": [],   # 다음 병렬 처리를 위해 초기화
    }


# 4. 그래프 노드들

def node_session_start(state: State) -> dict:
    return {"phase": "session_start", "parallel_results": [], "session_log": []}

def node_expression_select(state: State) -> dict:
    expression = state.get("expression", "")
    difficulty = check_difficulty.invoke({"expression": expression})
    synonyms   = find_synonyms.invoke({"expression": expression})
    st.session_state["difficulty"] = difficulty
    st.session_state["synonyms"]   = synonyms
    return {"phase": "expression_select", "expression": expression}

def node_reading_fetch(state: State) -> dict:
    expr      = state["expression"]
    article   = search_guardian.invoke({"expression": expr})
    sentences = article.get("sentences", [])
    sources   = article.get("sources",   [])
    while len(sentences) < 2: sentences.append("")
    while len(sources)   < 2: sources.append("Guardian")
    return {
        "phase"           : "reading",
        "article_title"   : " / ".join(s for s in sources if s),
        "article_date"    : "",
        "article_url"     : "",
        "article_body_en" : json.dumps({"sentences": sentences, "sources": sources}, ensure_ascii=False),
        "parallel_results": [],
    }

def route_parallel(state: State):
    return [
        Send("node_translate", state),
        Send("node_analyze",   state),
        Send("node_examples",  state),
    ]

def node_reading_merge(state: State) -> dict:
    results = {r["type"]: r["content"] for r in state["parallel_results"]}
    return {
        "phase"           : "reading_done",
        "article_body_kr" : results.get("kr",       "번역 없음"),
        "article_analysis": results.get("analysis", "분석 없음"),
        "article_examples": results.get("examples", "예문 없음"),
    }

def node_writing(state: State) -> dict:
    expr          = state["expression"]
    writing_input = state.get("writing_input", "")
    result = tutor_llm.invoke([   
        SystemMessage(content="You are a friendly Korean English tutor. "
                              "Give warm, encouraging feedback in Korean. "
                              "Always be constructive."),
        HumanMessage(content=f"""
학습 표현: '{expr}' / 유저 문장: '{writing_input}'
4항목 한국어 피드백:
1. ✅ 표현 사용 여부  2. 🔧 문법 구체적으로 체크  3. 💬 더 자연스러운 영어 표현 제공  4. ⭐ 총평
""")
    ])
    return {
        "phase"           : "writing",
        "writing_input"   : writing_input,
        "writing_feedback": result.content.strip(),
    }

    
def node_quiz_generate(state: State) -> dict:
    """퀴즈 문제 생성"""
    expr = state['expression']
    result = quiz_llm.invoke([   # 메시지 리스트로 전달
        SystemMessage(content="You are a strict IELTS examiner. "
                              "Create precise fill-in-the-blank questions. "
                              "Follow format instructions exactly."),
        HumanMessage(content=f"""
다음 조건을 반드시 지켜서 영어 빈칸 문제를 만들어라:

1. 학습 표현: '{expr}'
2. 문제 문장에서 '{expr}' 부분을 '______'로 반드시 대체하라
3. 보기 4개 중 정답은 반드시 '{expr}'이어야 한다
4. 나머지 보기 3개는 '{expr}'과 비슷하지만 틀린 표현으로 구성하라
5. IELTS B2 수준의 자연스러운 영어 문장을 사용하라

형식 엄수 (다른 말 절대 추가 금지):
Q. (______ 포함한 문장)
A. (보기1)
B. (보기2)
C. (보기3)
D. (보기4)
정답: (A/B/C/D)
""")
    ])
    lines    = [l.strip() for l in result.content.strip().split("\n") if l.strip()]
    question, choices, answer = "", [], ""
    for line in lines:
        if   line.startswith("Q."): question = line
        elif line[:2] in ("A.","B.","C.","D."): choices.append(line)
        elif line.startswith("정답:"): answer = line.replace("정답:","").strip()
    return {
        "phase"        : "quiz_ready",
        "quiz_question": question,
        "quiz_choices" : choices,
        "quiz_answer"  : answer,
        "quiz_input"   : "",
        "quiz_feedback": "",
        "quiz_retry"   : False,
    }

def node_quiz_grade(state: State) -> dict:
    answer     = state.get("quiz_answer", "")
    quiz_input = state.get("quiz_input", "").strip().upper()
    correct    = quiz_input == answer.upper()
    feedback   = f"✅ 정답! ({answer})" if correct else \
                 f"❌ 오답. 선택: {quiz_input} | 정답: {answer}"
    return {
        "phase"        : "quiz_graded",
        "quiz_feedback": feedback,
        "quiz_retry"   : not correct,
    }

def node_save(state: State) -> dict:
    log_entry = {
        "expression": state["expression"],
        "article"   : state["article_title"],
        "writing"   : state["writing_input"],
        "quiz"      : state["quiz_feedback"],
    }
    return {"phase": "done", "session_log": [log_entry]}

def route_after_grade(state: State) -> str:
    """퀴즈 채점 후 저장 여부 결정"""
    return "save" if not state.get("quiz_retry") else END


# 5. 그래프 조립

@st.cache_resource
def build_graphs():
    memory = MemorySaver()

    # Graph A: 리딩 파이프라인
    b_read = StateGraph(State)
    b_read.add_node("node_session_start",     node_session_start)
    b_read.add_node("orchestrator_node",      orchestrator_node)   # 추가
    b_read.add_node("worker_node",            worker_node)         # 추가
    b_read.add_node("node_workers_merge",     node_workers_merge)  # 추가
    b_read.add_node("node_translate",         node_translate)
    b_read.add_node("node_analyze",           node_analyze)
    b_read.add_node("node_examples",          node_examples)
    b_read.add_node("node_reading_merge",     node_reading_merge)
    b_read.set_entry_point("node_session_start")

    # node_session_start → orchestrator
    b_read.add_edge("node_session_start", "orchestrator_node")

    # orchestrator → workers (병렬)
    b_read.add_conditional_edges(
        "orchestrator_node", route_workers, ["worker_node"]
    )

    # workers → merge
    b_read.add_edge("worker_node", "node_workers_merge")

    # merge → 번역/분석/예문 병렬
    b_read.add_conditional_edges(
        "node_workers_merge", route_parallel,
        ["node_translate", "node_analyze", "node_examples"]
    )
    b_read.add_edge("node_translate",     "node_reading_merge")
    b_read.add_edge("node_analyze",       "node_reading_merge")
    b_read.add_edge("node_examples",      "node_reading_merge")
    b_read.add_edge("node_reading_merge", END)
    graph_read = b_read.compile(checkpointer=memory)

    # Graph B: 쓰기 피드백
    b_write = StateGraph(State)
    b_write.add_node("node_writing", node_writing)
    b_write.set_entry_point("node_writing")
    b_write.add_edge("node_writing", END)
    graph_write = b_write.compile()

    # Graph C: 퀴즈 생성
    b_quiz_gen = StateGraph(State)
    b_quiz_gen.add_node("node_quiz_generate", node_quiz_generate)
    b_quiz_gen.set_entry_point("node_quiz_generate")
    b_quiz_gen.add_edge("node_quiz_generate", END)
    graph_quiz_gen = b_quiz_gen.compile()

    # Graph D: 퀴즈 채점 + 저장
    b_quiz_grade = StateGraph(State)
    b_quiz_grade.add_node("node_quiz_grade", node_quiz_grade)
    b_quiz_grade.add_node("node_save",       node_save)
    b_quiz_grade.set_entry_point("node_quiz_grade")
    b_quiz_grade.add_conditional_edges(
        "node_quiz_grade",
        route_after_grade,
        {"save": "node_save", END: END}
    )
    b_quiz_grade.add_edge("node_save", END)
    graph_quiz_grade = b_quiz_grade.compile()

    return graph_read, graph_write, graph_quiz_gen, graph_quiz_grade

graph_read, graph_write, graph_quiz_gen, graph_quiz_grade = build_graphs()


# 6. Streamlit UI

st.set_page_config(page_title="🇬🇧 영어 학습 앱", page_icon="📚", layout="wide")
st.title("📚 LangGraph 영어 학습 앱")

defaults = {
    "step"            : "input_expression",
    "state"           : {},
    "thread_id"       : "session-001",
    "quiz_choices"    : [],
    "quiz_question"   : "",
    "quiz_answer"     : "",
    "quiz_retry"      : False,
    "difficulty"      : "",
    "synonyms"        : "",
    "writing_input"   : "",
    "writing_feedback": "",   # 추가: 피드백 별도 보관
    "writing_done"    : False, # 추가: 피드백 완료 여부
    "quiz_graded"     : False,   # 추가
    "quiz_retry"      : False,   # 추가
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# 사이드바
with st.sidebar:
    st.header("📋 학습 기록")
    logs = st.session_state["state"].get("session_log", [])
    if logs:
        for i, log in enumerate(logs, 1):
            with st.expander(f"세션 {i}: {log.get('expression','')}"):
                st.write(f"📰 기사: {log.get('article','')}")
                st.write(f"✍️ 내 문장: {log.get('writing','')}")
                st.write(f"📝 퀴즈: {log.get('quiz','')}")
    else:
        st.info("아직 완료된 세션이 없습니다.")
    if st.button("🔄 새 세션 시작"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# STEP 1: 표현 입력

if st.session_state["step"] == "input_expression":
    st.subheader("① 오늘 배울 영어 표현을 입력하세요")
    with st.form("form_expression"):
        expression = st.text_input("✏️ 표현 입력", placeholder="예: stream")
        submitted  = st.form_submit_button("🚀 학습 시작")
    if submitted and expression.strip():
        init_state: State = {
            "phase": "", "expression": expression.strip(),
            "article_title": "", "article_date": "", "article_url": "",
            "article_body_en": "", "article_body_kr": "",
            "article_analysis": "", "article_examples": "",
            "writing_input": "", "writing_feedback": "",
            "quiz_question": "", "quiz_choices": [], "quiz_answer": "",
            "quiz_input": "", "quiz_feedback": "", "quiz_retry": False,
            "parallel_results": [], "session_log": [],
            "orchestrator_tasks": [],
            "current_task"      : "",
        }
        st.info("⏳ AI가 아래 단계를 순서대로 처리하고 있어요!")
        progress = st.progress(0, text="🔍 표현 난이도 분석 중...")
        with st.spinner("🔍 분석 중... 잠시만 기다려주세요!"):
            try:
                progress.progress(25, text="📰 Guardian 기사 검색 중...")
                time.sleep(0.5)
                progress.progress(50, text="🇰🇷 번역 및 분석 중...")
                time.sleep(0.5)
                progress.progress(75, text="📝 예문 생성 중...")
                result = graph_read.invoke(init_state, config)
                progress.progress(100, text="✅ 완료!")
                st.session_state["state"] = result
                st.session_state["step"]  = "show_reading"
            except Exception as e:
                st.error("⚠️ 기사 검색 중 오류가 발생했어요. 잠시 후 다시 시도해주세요!")
                st.caption(f"오류 상세: {e}")
        st.rerun()


# STEP 2: 기사 & 번역 & 분석 표시

elif st.session_state["step"] == "show_reading":
    s    = st.session_state["state"]
    expr = s.get("expression", "")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📌 학습 표현", expr)
        st.metric("📊 난이도",   st.session_state.get("difficulty", "-"))
    with col2:
        st.metric("🔄 유의어",   st.session_state.get("synonyms", "-"))

    st.divider()

    try:
        body_data = json.loads(s.get("article_body_en", "{}"))
        sentences = body_data.get("sentences", [])
    except Exception:
        sentences = []

    tab1, tab2, tab3, tab4 = st.tabs(["📄 원문", "🇰🇷 번역", "🔍 분석", "📝 예문"])
    with tab1:
        st.subheader(f"📰 {s.get('article_title','')}")
        for i, sent in enumerate(sentences, 1):
            st.markdown(f"**{i}.** {sent}")
    with tab2:
        st.markdown(s.get("article_body_kr", ""))
    with tab3:
        st.markdown(s.get("article_analysis", ""))
    with tab4:
        st.markdown(s.get("article_examples", ""))

    if st.button("✍️ 쓰기 연습으로 이동 →"):
        st.session_state["writing_done"]     = False   # 초기화
        st.session_state["writing_feedback"] = ""
        st.session_state["step"] = "writing"
        st.rerun()


# STEP 3: 쓰기 연습  ★ 핵심 수정 부분 ★

elif st.session_state["step"] == "writing":
    s    = st.session_state["state"]
    expr = s.get("expression", "")
    st.subheader(f"✍️ 쓰기 연습  |  📌 표현: **{expr}**")

    # 아직 피드백 전: 입력 폼 표시
    if not st.session_state["writing_done"]:
        # 참고 예문 표시
        examples = st.session_state["state"].get("article_examples", "")
        if examples:
            with st.expander("💡 참고 예문 보기 (클릭해서 펼치기)"):
                st.markdown(examples)

        with st.form("form_writing"):
            writing_input = st.text_area(
                "표현을 사용한 영어 문장을 작성하세요",
                placeholder=f"'{expr}'을 포함한 문장을 써보세요..."
            )
            submitted = st.form_submit_button("📤 제출")

        if submitted and writing_input.strip():
            st.info("✏️ AI 튜터가 문장을 꼼꼼히 읽고 있어요!")
            with st.spinner("💬 피드백 작성 중... 10초 내로 완료돼요!"):
                try:
                    new_state = {**s, "writing_input": writing_input.strip()}
                    result    = graph_write.invoke(new_state)
                    st.session_state["state"]            = {**s, **result}
                    st.session_state["writing_feedback"] = result.get("writing_feedback", "")
                    st.session_state["writing_input"]    = writing_input.strip()
                    st.session_state["writing_done"]     = True
                except Exception as e:
                    st.error("⚠️ 피드백 생성 중 오류가 발생했어요. 다시 제출해보세요!")
                    st.caption(f"오류 상세: {e}")
            st.rerun()

    # 피드백 완료 후: 내 문장 + 피드백 함께 표시
    if st.session_state["writing_done"]:
        # ① 내가 쓴 문장 먼저
        st.markdown("#### ✏️ 내가 쓴 문장")
        st.info(st.session_state["writing_input"])   # 저장된 문장 표시

        # ② 피드백
        st.markdown("#### 💡 피드백")
        st.success(st.session_state["writing_feedback"])

        st.divider()
        if st.button("❓ 퀴즈로 이동 →", type="primary"):
            st.session_state["step"] = "quiz"
            st.rerun()


# STEP 4: 퀴즈

elif st.session_state["step"] == "quiz":
    s = st.session_state["state"]

    # 퀴즈 문제 생성
    if not s.get("quiz_question"):
        st.info("🧠 학습한 표현으로 퀴즈를 만들고 있어요!")
        with st.spinner("📝 퀴즈 생성 중... 잠깐만요!"):
            try:
                result = graph_quiz_gen.invoke(s)
                st.session_state["state"] = {**s, **result}
                s = st.session_state["state"]
            except Exception as e:
                st.error("⚠️ 퀴즈 생성에 실패했어요. 페이지를 새로고침 해주세요!")
                st.caption(f"오류 상세: {e}")
                st.stop()

    st.subheader("❓ 퀴즈")
    st.markdown(f"**{s.get('quiz_question','')}**")
    choices = s.get("quiz_choices", [])

    if not choices:
        st.error("퀴즈 생성에 실패했습니다. 새로고침 해주세요.")

    else:
        # 아직 채점 전: 문제 폼 표시
        if not st.session_state.get("quiz_graded"):
            choice_map = {c[0]: c for c in choices}
            with st.form("form_quiz"):
                selected  = st.radio(
                    "정답을 선택하세요",
                    options=[c[0] for c in choices],
                    format_func=lambda x: choice_map.get(x, x)
                )
                submitted = st.form_submit_button("✅ 제출")

            if submitted:
                st.info("🎯 답안을 채점하고 있어요!")
                with st.spinner("✅ 채점 중..."):
                    try:
                        graded = graph_quiz_grade.invoke({**s, "quiz_input": selected})
                        st.session_state["state"]       = {**s, **graded}
                        st.session_state["quiz_graded"] = True
                        st.session_state["quiz_retry"]  = graded.get("quiz_retry", False)
                    except Exception as e:
                        st.error("⚠️ 채점 중 오류가 발생했어요. 다시 제출해보세요!")
                        st.caption(f"오류 상세: {e}")
                st.rerun()  # rerun으로 블록 밖에서 결과 렌더링

        # 채점 완료 후: 결과 표시 (블록 밖!)
        if st.session_state.get("quiz_graded"):
            feedback = st.session_state["state"].get("quiz_feedback", "")

            if st.session_state.get("quiz_retry"):
                st.error(feedback)
                st.warning("🔄 다시 도전해보세요!")
                if st.button("🔄 다시 풀기"):
                    # 퀴즈 상태 초기화 (문제는 유지)
                    st.session_state["quiz_graded"] = False
                    st.session_state["quiz_retry"]  = False
                    st.session_state["state"]["quiz_input"] = ""
                    st.rerun()
            else:
                st.success(feedback)
                if st.button("🎉 결과 저장 →", type="primary"):   # 블록 밖!
                    st.session_state["quiz_graded"] = False
                    st.session_state["step"]        = "done"
                    st.rerun()


# STEP 5: 완료

elif st.session_state["step"] == "done":
    s = st.session_state["state"]
    st.balloons()
    st.success("🎉 세션 완료!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📌 학습 표현",  s.get("expression",""))
        st.metric("📝 퀴즈 결과", s.get("quiz_feedback",""))
    with col2:
        st.metric("✍️ 내 문장",   s.get("writing_input",""))
        logs = s.get("session_log", [])
        st.metric("📚 누적 세션",  f"{len(logs)}회")

    st.divider()
    st.subheader("📋 전체 학습 요약")
    for i, log in enumerate(logs, 1):
        with st.expander(f"세션 {i}: {log.get('expression','')}"):
            st.json(log)

    if st.button("🔄 새 표현 학습하기"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()