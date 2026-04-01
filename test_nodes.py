import os, json, pytest
os.environ["OPENAI_API_KEY"]   = "your-key"
os.environ["GUARDIAN_API_KEY"] = "your-key"

from main import node_writing, node_quiz_generate, node_quiz_grade, State
from langchain_openai import ChatOpenAI

@pytest.fixture
def base_state() -> State:
    return {
        "phase": "", "expression": "break the ice",
        "article_title": "Test", "article_date": "", "article_url": "",
        "article_body_en": json.dumps({
            "sentences": [
                "She broke the ice with a funny joke.",
                "It helped break the ice at the meeting."
            ]
        }),
        "article_body_kr": "", "article_analysis": "", "article_examples": "",
        "writing_input": "", "writing_feedback": "",
        "quiz_question": "", "quiz_choices": [], "quiz_answer": "",
        "quiz_input": "", "quiz_feedback": "", "quiz_retry": False,
        "parallel_results": [], "session_log": [],
        "orchestrator_tasks": [], "current_task": "",
    }

# 노드 단위 테스트
def test_writing_feedback_not_empty(base_state):
    """쓰기 피드백이 비어있지 않아야 한다"""
    state  = {**base_state, "writing_input": "I broke the ice at the party."}
    result = node_writing(state)
    assert result["writing_feedback"] != "", "피드백이 비어있음"
    assert result["phase"] == "writing"

def test_quiz_has_blank(base_state):
    """퀴즈 문제에 빈칸(______)이 있어야 한다"""
    result = node_quiz_generate(base_state)
    assert "______" in result["quiz_question"], "빈칸 없음"
    assert len(result["quiz_choices"]) == 4,    "보기가 4개가 아님"
    assert result["quiz_answer"] in ["A","B","C","D"], "정답 형식 오류"

def test_quiz_grade_correct(base_state):
    """정답 입력 시 quiz_retry == False"""
    state  = {**base_state, "quiz_answer": "A", "quiz_input": "A"}
    result = node_quiz_grade(state)
    assert result["quiz_retry"]   == False
    assert "✅" in result["quiz_feedback"]

def test_quiz_grade_wrong(base_state):
    """오답 입력 시 quiz_retry == True"""
    state  = {**base_state, "quiz_answer": "A", "quiz_input": "B"}
    result = node_quiz_grade(state)
    assert result["quiz_retry"]  == True
    assert "❌" in result["quiz_feedback"]

# AI-as-Judge
def test_quiz_quality_ai_judge(base_state):
    """AI Judge가 퀴즈 품질 3점 이상 평가"""
    judge_llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                             api_key=os.environ["OPENAI_API_KEY"])
    quiz_result = node_quiz_generate(base_state)
    question    = quiz_result["quiz_question"]
    choices     = "\n".join(quiz_result["quiz_choices"])

    verdict = judge_llm.invoke(f"""
다음 영어 퀴즈를 평가하고 1~5 숫자만 출력하라 (5=매우좋음).
평가기준:
- 빈칸(______)이 있는가
- 보기가 4개인가
- 문장이 자연스러운가
- 정답이 학습 표현인가

문제: {question}
보기:
{choices}

숫자만 출력:""")

    score = int(verdict.content.strip())
    assert score >= 3, f"퀴즈 품질 미달: {score}/5점"