import asyncio
import time
import uuid
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .prompts import (
    ANSWER_PROMPT,
    CHECK_AND_REWRITE_PROMPT,
    COMMON_POLICY,
    IMAGE_SUMMARY_PROMPT,
)
from .rag import format_answer_with_image_list, format_contexts, retrieve


NO_IMAGE_SUMMARY = "无"
GENERAL_POLICY_CONTEXT = "通用客服政策题：优先使用通用客服政策参考，不引用无关商品手册。"
CHAT_MAX_TOKENS = 450
_OUTPUT_PARSER = StrOutputParser()
GENERAL_POLICY_KEYWORDS = (
    "7天",
    "七天",
    "无理由",
    "退换货",
    "退货",
    "换货",
    "退款",
    "退回",
    "运费",
    "邮费",
    "快递费",
    "发票",
    "开票",
    "抬头",
    "投诉",
    "颜色偏差",
    "和图片不一样",
    "物流",
    "待揽收",
    "发货",
    "送到",
    "乡镇",
    "收货",
    "订单",
    "破损",
    "少件",
    "错发",
    "漏发",
    "补发",
    "return",
    "refund",
    "shipping",
    "invoice",
    "complaint",
    "delivery",
    "logistics",
    "wrong item",
    "missing item",
)
GENERAL_REPAIR_KEYWORDS = (
    "售后维修",
    "维修服务",
    "服务范围",
    "人为损坏",
    "维修费用",
    "维修费",
    "保修",
    "质保",
    "warranty",
)


class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    image_summary: str
    contexts: str
    answer: str


def answer_question(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str]:
    sid = session_id or f"kf_session_{uuid.uuid4().hex}"
    input_state = {
        "question": question,
        "images": images or [],
        "session_id": sid,
    }
    if contexts is not None:
        input_state["contexts"] = contexts
    state = get_graph().invoke(input_state)
    return state["answer"], state["session_id"]


async def answer_question_async(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str]:
    return await asyncio.to_thread(
        answer_question,
        question,
        images=images,
        session_id=session_id,
        contexts=contexts,
    )


def get_graph():
    builder = StateGraph(AgentState)
    builder.add_node("summarize_images", summarize_images)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("check_answer", check_answer)

    builder.add_edge(START, "summarize_images")
    builder.add_edge("summarize_images", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", "check_answer")
    builder.add_edge("check_answer", END)
    return builder.compile()


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
                "text": IMAGE_SUMMARY_PROMPT,
            }
        ]
        for image in images[:3]:
            content.append({"type": "image_url", "image_url": {"url": image}})
        model = init_chat_model(
            model=settings.vision_model,
            model_provider="openai",
            api_key=settings.openai_api_key,
            base_url=settings.vision_base_url,
            temperature=0,
            timeout=settings.model_timeout_seconds,
            max_retries=1,
        )
        state["image_summary"] = _OUTPUT_PARSER.invoke(
            model.invoke([HumanMessage(content=content)])
        ).strip()
    except Exception:
        state["image_summary"] = (
            "图片解析失败；请结合文字问题回答，必要时要求用户补充图片信息。"
        )
    return state


def retrieve_context(state: AgentState) -> AgentState:
    if _is_general_policy_question(state["question"]):
        state["contexts"] = GENERAL_POLICY_CONTEXT
        return state

    if "contexts" in state:
        return state

    query = state["question"]
    image_summary = state.get("image_summary")
    if image_summary and image_summary != NO_IMAGE_SUMMARY:
        query = f"{query}\n{image_summary}"
    state["contexts"] = format_contexts(retrieve(query))
    return state


def _is_general_policy_question(question: str) -> bool:
    normalized = question.lower()
    if any(keyword in normalized for keyword in GENERAL_POLICY_KEYWORDS):
        return True
    return any(keyword in normalized for keyword in GENERAL_REPAIR_KEYWORDS)


def generate_answer(state: AgentState) -> AgentState:
    _require_chat_model()
    prompt = ANSWER_PROMPT.format(
        image_summary=state.get("image_summary", NO_IMAGE_SUMMARY),
        contexts=state.get("contexts", ""),
        common_policy=COMMON_POLICY,
        question=state["question"],
    )

    answer = _invoke_chat(prompt, error_context="generate answer")
    if not answer:
        answer = _check_and_rewrite_answer(state, "")
    if not answer:
        raise RuntimeError("chat model returned an empty answer")

    state["answer"] = answer
    return state


def check_answer(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    final_answer = _check_and_rewrite_answer(state, answer)
    if not final_answer:
        raise RuntimeError("check and rewrite returned an empty answer")

    state["answer"] = format_answer_with_image_list(
        final_answer, state.get("contexts", "")
    )
    return state


def _check_and_rewrite_answer(state: AgentState, answer: str) -> str:
    prompt = CHECK_AND_REWRITE_PROMPT.format(
        image_summary=state.get("image_summary", NO_IMAGE_SUMMARY),
        contexts=state.get("contexts", ""),
        common_policy=COMMON_POLICY,
        question=state["question"],
        answer=answer,
    )
    return _invoke_chat(prompt, error_context="check and rewrite answer")


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


def _invoke_chat(prompt: str, error_context: str) -> str:
    settings = get_settings()
    try:
        model = init_chat_model(
            model=settings.chat_model,
            model_provider="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.2,
            max_tokens=CHAT_MAX_TOKENS,
            timeout=settings.model_timeout_seconds,
            max_retries=1,
        )
        return _OUTPUT_PARSER.invoke(model.invoke([HumanMessage(content=prompt)])).strip()
    except Exception as exc:
        raise RuntimeError(f"failed to {error_context} with chat model") from exc


def _require_chat_model() -> None:
    if not get_settings().has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be configured so the chat model can generate answers."
        )
