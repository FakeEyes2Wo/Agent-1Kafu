import time
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .memory import load_history, save_turn
from .prompts import ANSWER_PROMPT, COMMON_POLICY, REFLECTION_PROMPT, REWRITE_PROMPT
from .rag import format_contexts, retrieve


NO_IMAGE_SUMMARY = "无"


class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    persist_history: bool
    history: list[dict[str, str]]
    image_summary: str
    contexts: str
    answer: str


def answer_question(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    persist_history: bool = True,
) -> tuple[str, str]:
    sid = session_id or f"kf_session_{uuid.uuid4().hex}"
    state = get_graph().invoke(
        {
            "question": question,
            "images": images or [],
            "session_id": sid,
            "persist_history": persist_history,
        }
    )
    return state["answer"], state["session_id"]


def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("load_context", load_context)
    builder.add_node("summarize_images", summarize_images)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("check_answer", check_answer)
    builder.add_node("rewrite_answer", rewrite_answer)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "summarize_images")
    builder.add_edge("summarize_images", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", "check_answer")
    builder.add_edge("check_answer", "rewrite_answer")
    builder.add_edge("rewrite_answer", "save_memory")
    builder.add_edge("save_memory", END)
    return builder.compile()


def load_context(state: AgentState) -> AgentState:
    if state.get("persist_history", True):
        state["history"] = load_history(state["session_id"])
    else:
        state["history"] = []
    return state


def summarize_images(state: AgentState) -> AgentState:
    images = state.get("images") or []
    if not images:
        state["image_summary"] = NO_IMAGE_SUMMARY
        return state

    settings = get_settings()
    if not settings.has_openai_key:
        state["image_summary"] = "用户上传了图片，但当前未配置视觉模型；请结合文字问题回答。"
        return state

    try:
        content = [
            {
                "type": "text",
                "text": "请简要提取这些客服图片中的订单、物流、故障或商品信息。",
            }
        ]
        for image in images[:3]:
            content.append({"type": "image_url", "image_url": {"url": image}})
        llm = _chat_llm(model=settings.vision_model, base_url=settings.vision_base_url)
        state["image_summary"] = str(llm.invoke([HumanMessage(content=content)]).content)
    except Exception:
        state["image_summary"] = (
            "图片解析失败；请结合文字问题回答，必要时要求用户补充图片信息。"
        )
    return state


def retrieve_context(state: AgentState) -> AgentState:
    query = state["question"]
    image_summary = state.get("image_summary")
    if image_summary and image_summary != NO_IMAGE_SUMMARY:
        query = f"{query}\n{image_summary}"
    state["contexts"] = format_contexts(retrieve(query))
    return state


def generate_answer(state: AgentState) -> AgentState:
    _require_chat_model()
    history = _format_history(state.get("history", []))
    prompt = ANSWER_PROMPT.format(
        history=history,
        image_summary=state.get("image_summary", NO_IMAGE_SUMMARY),
        contexts=state.get("contexts", ""),
        common_policy=COMMON_POLICY,
        question=state["question"],
    )

    answer = _invoke_chat(prompt, error_context="generate answer")
    if not answer:
        answer = _reflect_answer(state, "")
    if not answer:
        raise RuntimeError("chat model returned an empty answer")

    state["answer"] = answer
    return state


def check_answer(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    reflected = _reflect_answer(state, answer)
    if not reflected:
        raise RuntimeError("reflection returned an empty answer")

    state["answer"] = reflected
    return state


def rewrite_answer(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    rewritten = _rewrite_answer(state, answer)
    if rewritten:
        state["answer"] = rewritten
    return state


def _reflect_answer(state: AgentState, answer: str) -> str:
    history = _format_history(state.get("history", []))
    prompt = REFLECTION_PROMPT.format(
        history=history,
        image_summary=state.get("image_summary", NO_IMAGE_SUMMARY),
        contexts=state.get("contexts", ""),
        common_policy=COMMON_POLICY,
        question=state["question"],
        answer=answer,
    )
    return _invoke_chat(prompt, error_context="reflect answer")


def _rewrite_answer(state: AgentState, answer: str) -> str:
    history = _format_history(state.get("history", []))
    prompt = REWRITE_PROMPT.format(
        history=history,
        image_summary=state.get("image_summary", NO_IMAGE_SUMMARY),
        contexts=state.get("contexts", ""),
        common_policy=COMMON_POLICY,
        question=state["question"],
        answer=answer,
    )
    return _invoke_chat(prompt, error_context="rewrite answer")


def save_memory(state: AgentState) -> AgentState:
    if state.get("persist_history", True):
        save_turn(state["session_id"], state["question"], state["answer"])
    return state


def response_payload(answer: str, session_id: str) -> dict:
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
        },
    }


def _chat_llm(model: str, base_url: str, temperature: float = 0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=settings.model_timeout_seconds,
        max_retries=1,
    )


def _invoke_chat(prompt: str, error_context: str) -> str:
    settings = get_settings()
    try:
        llm = _chat_llm(
            model=settings.chat_model,
            base_url=settings.openai_base_url,
            temperature=0.2,
        )
        return str(llm.invoke(prompt).content).strip()
    except Exception as exc:
        raise RuntimeError(f"failed to {error_context} with chat model") from exc


def _require_chat_model() -> None:
    if not get_settings().has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be configured so the chat model can generate answers."
        )


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "无"
    lines = []
    for turn in history:
        lines.append(
            f"用户：{turn.get('question', '')}\n客服：{turn.get('answer', '')}"
        )
    return "\n".join(lines)
