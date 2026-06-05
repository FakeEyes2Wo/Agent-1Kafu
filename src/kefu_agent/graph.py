import asyncio
import re
import time
import uuid
from typing import Any, TypedDict

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
    "虚假宣传",
    "质量问题",
    "瑕疵",
    "划痕",
    "颜色偏差",
    "和图片不一样",
    "正品",
    "假货",
    "二手",
    "污渍",
    "拆封",
    "临期",
    "保质期",
    "生产日期",
    "受潮",
    "赔偿",
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
    "签收",
    "取消订单",
    "优惠券",
    "纸质版说明书",
    "电子版",
    "说明书",
    "售后保障卡",
    "以旧换新",
    "智能客服",
    "上门安装",
    "安装人员",
    "配件费",
    "试用装",
    "试用期间",
    "国际配送",
    "国外",
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
    "寄回维修",
    "上门检修",
    "终身维修",
    "保修",
    "质保",
    "质保期",
    "warranty",
)


class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    image_summary: str
    contexts: str
    draft_answer: str
    checked_answer: str
    answer: str
    api_calls: list[dict[str, Any]]


class ChatText(str):
    def __new__(cls, value: str, usage: dict[str, int] | None = None):
        item = str.__new__(cls, value)
        item.usage = usage or {}
        return item


def answer_question(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str]:
    answer, sid, _trace = answer_question_with_trace(
        question,
        images=images,
        session_id=session_id,
        contexts=contexts,
    )
    return answer, sid


def answer_question_with_trace(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    sid = session_id or f"kf_session_{uuid.uuid4().hex}"
    input_state = {
        "question": question,
        "images": images or [],
        "session_id": sid,
    }
    if contexts is not None:
        input_state["contexts"] = contexts
    state = get_graph().invoke(input_state)
    return state["answer"], state["session_id"], _trace_from_state(state)


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


async def answer_question_with_trace_async(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    return await asyncio.to_thread(
        answer_question_with_trace,
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
            api_key=settings.model_api_key,
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
    _record_api_call(state, "generate_answer", answer)
    if not answer:
        answer = _check_and_rewrite_answer(state, "")
    if not answer:
        answer = _fallback_answer_from_contexts(state)

    state["draft_answer"] = str(answer)
    state["answer"] = answer
    return state


def check_answer(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    state.setdefault("draft_answer", answer)
    final_answer = _check_and_rewrite_answer(state, answer)
    if not final_answer:
        final_answer = answer
    if not final_answer:
        raise RuntimeError("check and rewrite returned an empty answer")

    state["checked_answer"] = str(final_answer)
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
    answer = _invoke_chat(prompt, error_context="check and rewrite answer")
    _record_api_call(state, "check_and_rewrite_answer", answer)
    return answer


def _fallback_answer_from_contexts(state: AgentState) -> str:
    snippet = _first_context_snippet(state.get("contexts", ""))
    if snippet:
        if _looks_english(state["question"]):
            return f"According to the available manual information, {snippet}"
        return f"根据说明，{snippet}"
    if _looks_english(state["question"]):
        return "The available evidence is not enough to confirm the exact step. Please provide the product model, symptoms, or image details for further checking."
    return "目前资料不足以确认具体处理步骤，请补充商品型号、故障现象或图片细节后再核实处理。"


def _first_context_snippet(contexts: str, max_chars: int = 160) -> str:
    for block in contexts.split("\n\n"):
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if not line or re.match(r"^\[\d+\]\s", line):
                continue
            if re.match(r"^(可用图片|image_ids|images)[:：]", line, flags=re.I):
                continue
            lines.append(line)
        text = " ".join(lines)
        text = re.sub(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", "<PIC>", text, flags=re.I)
        text = re.sub(r"(?:可用图片|image_ids|images)[:：]\s*\[[^\[\]]*\]", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return text[:max_chars].rstrip()
    return ""


def _looks_english(question: str) -> bool:
    return bool(re.search(r"[A-Za-z]", question)) and not re.search(r"[\u4e00-\u9fff]", question)


def _chat_extra_body(settings: Any) -> dict[str, Any] | None:
    source = getattr(settings, "model_api_key_source", "")
    base_url = str(getattr(settings, "model_base_url", "")).lower()
    if source not in {"bailian", "dashscope"} and "dashscope" not in base_url:
        return None
    return {"enable_thinking": bool(getattr(settings, "chat_enable_thinking", False))}


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
            api_key=settings.model_api_key,
            base_url=settings.model_base_url,
            temperature=0.2,
            max_tokens=CHAT_MAX_TOKENS,
            timeout=settings.model_timeout_seconds,
            max_retries=1,
            extra_body=_chat_extra_body(settings),
        )
        message = model.invoke([HumanMessage(content=prompt)])
        answer = _OUTPUT_PARSER.invoke(message).strip()
        return ChatText(answer, _usage_from_message(message))
    except Exception as exc:
        raise RuntimeError(f"failed to {error_context} with chat model") from exc


def _usage_from_message(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return {
        str(key): int(value)
        for key, value in dict(usage).items()
        if isinstance(value, int | float)
    }


def _record_api_call(state: AgentState, phase: str, answer: str) -> None:
    calls = list(state.get("api_calls") or [])
    calls.append(
        {
            "phase": phase,
            "usage": getattr(answer, "usage", {}) or {},
            "text_chars": len(str(answer)),
        }
    )
    state["api_calls"] = calls


def _trace_from_state(state: AgentState) -> dict[str, Any]:
    api_calls = list(state.get("api_calls") or [])
    return {
        "image_summary": state.get("image_summary", ""),
        "contexts": state.get("contexts", ""),
        "draft_answer": state.get("draft_answer", ""),
        "checked_answer": state.get("checked_answer", ""),
        "final_answer": state.get("answer", ""),
        "api_calls": api_calls,
        "usage": _merge_api_usage(api_calls),
    }


def _merge_api_usage(api_calls: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for call in api_calls:
        usage = call.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int | float):
                merged[str(key)] = merged.get(str(key), 0) + int(value)
    return merged


def _require_chat_model() -> None:
    if not get_settings().has_openai_key:
        raise RuntimeError(
            "BAILIAN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY must be configured "
            "so the chat model can generate answers."
        )
