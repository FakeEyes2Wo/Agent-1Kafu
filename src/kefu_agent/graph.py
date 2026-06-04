import asyncio
import re
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
    ANSWER_VERIFICATION_PROMPT,
    CHECK_AND_REWRITE_PROMPT,
    COMMON_POLICY,
    IMAGE_SUMMARY_PROMPT,
)
from .rag import format_answer_with_image_list, format_contexts, retrieve


NO_IMAGE_SUMMARY = "无"
GENERAL_POLICY_CONTEXT = "通用客服政策题：优先使用通用客服政策参考，不引用无关手册。"
CHAT_MAX_TOKENS = 700
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
MANUAL_PRODUCT_HINTS = (
    "吹风机",
    "空调",
    "遥控器",
    "蒸汽清洁机",
    "人体工学椅",
    "椅子",
    "洗碗机",
    "空气净化器",
    "健身单车",
    "电钻",
    "dcb",
    "健身追踪器",
    "冰箱",
    "冰柜",
    "发电机",
    "摩托艇",
    "水泵",
    "温控器",
    "vr头显",
    "vr 头显",
    "功能键盘",
    "儿童电动摩托车",
    "蓝牙激光鼠标",
    "烤箱",
    "相机",
    "airfryer",
    "boat",
    "ship steers",
    "sailing",
    "coffee machine",
    "coffee maker",
    "camera",
    "earphones",
    "ereader",
    "fax",
    "grill",
    "jetski",
    "jet ski",
    "landline",
    "lawn mower",
    "microwave",
    "motherboard",
)
MANUAL_REQUEST_HINTS = (
    "根据手册",
    "手册中",
    "手册说明",
    "如何",
    "怎么",
    "哪些步骤",
    "步骤",
    "前五条",
    "前六个",
    "最后三个",
    "组成部件",
    "核心部件",
    "按键",
    "指示灯",
    "闪烁",
    "标识",
    "安装",
    "拆卸",
    "更换",
    "清洁",
    "启动",
    "关闭",
    "充电",
    "调节",
    "设置",
    "连接",
    "使用",
    "操作",
    "保修包含",
    "保修政策",
    "三年有限保修",
    "warranty policy",
    "how to",
    "what are",
    "what should",
    "steps",
    "install",
    "remove",
    "replace",
    "clean",
    "start",
    "turn on",
    "turn off",
    "charge",
    "set",
    "connect",
    "operate",
)


class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    image_summary: str
    contexts: str
    response_plan: str
    answer: str
    verification_feedback: str


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
    builder.add_node("plan_answer", plan_answer)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("check_answer", check_answer)

    builder.add_edge(START, "summarize_images")
    builder.add_edge("summarize_images", "retrieve_context")
    builder.add_edge("retrieve_context", "plan_answer")
    builder.add_edge("plan_answer", "generate_answer")
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
        model = _init_openai_model(
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            temperature=0,
        )
        state["image_summary"] = _OUTPUT_PARSER.invoke(
            model.invoke([HumanMessage(content=_image_summary_content(images))])
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
    if _is_manual_question(normalized):
        return False
    return any(
        keyword in normalized
        for keyword in GENERAL_POLICY_KEYWORDS + GENERAL_REPAIR_KEYWORDS
    )


def _is_manual_question(normalized_question: str) -> bool:
    has_product = any(keyword in normalized_question for keyword in MANUAL_PRODUCT_HINTS)
    has_manual_request = any(
        keyword in normalized_question for keyword in MANUAL_REQUEST_HINTS
    )
    has_model = bool(re.search(r"\b[a-z]{1,8}\d{2,}[a-z0-9-]*\b", normalized_question))
    return (has_product and has_manual_request) or has_model


def plan_answer(state: AgentState) -> AgentState:
    state["response_plan"] = _build_response_plan(state)
    return state


def _build_response_plan(state: AgentState) -> str:
    question = state["question"]
    contexts = state.get("contexts", "")
    language = _customer_language(question)
    segments = _question_segments(question)
    is_policy = contexts.strip() == GENERAL_POLICY_CONTEXT
    has_context = bool(contexts.strip()) and not is_policy
    has_images = "<PIC>" in contexts or "可用图片" in contexts

    plan = [
        f"回答语言：{language}",
        f"问题类型：{'通用客服政策' if is_policy else '手册问答' if has_context else '证据不足'}",
        f"子问题数量：{len(segments)}；必须按原始顺序覆盖。",
    ]
    if is_policy:
        plan.append(
            "策略：直接给可执行客服处理口径；不要引用无关手册、型号、配件、授权经销商或无证据保修期。"
        )
    elif has_context:
        plan.append(
            "策略：优先使用排名靠前的手册证据，提取具体步骤、部件、条件、数值和限制；不要泛泛让用户查看说明书。"
        )
    else:
        plan.append(
            "策略：说明当前证据不足，只询问最少必要补充信息，避免编造具体参数、步骤或承诺。"
        )
    if has_images:
        plan.append(
            "图片策略：证据含配图时，在对应步骤、部件、状态或图示说明旁输出裸 <PIC>，不要输出图片ID。"
        )
    else:
        plan.append("图片策略：无可用配图证据时不要输出 <PIC>。")
    plan.append("终稿要求：短句、少客套、无Markdown标题、无内部检查说明。")
    return "\n".join(plan)


def _customer_language(question: str) -> str:
    return "中文" if any("\u4e00" <= ch <= "\u9fff" for ch in question) else "English"


def _question_segments(question: str) -> list[str]:
    normalized = question.replace("\r\n", "\n").replace("\r", "\n")
    pieces = re.split(r"\n+|[？?]\s*|[。!！]\s*", normalized)
    return [piece.strip(" \"'，,；;") for piece in pieces if piece.strip(" \"'，,；;")]


def generate_answer(state: AgentState) -> AgentState:
    _require_chat_model()
    prompt = ANSWER_PROMPT.format(**_prompt_inputs(state))

    answer = _invoke_chat(prompt, error_context="generate answer")
    if not answer:
        answer = _check_and_rewrite_answer(state, "")
    if not answer:
        raise RuntimeError("chat model returned an empty answer")

    state["answer"] = answer
    return state


def check_answer(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    feedback = _verify_answer_support(state, answer)
    state["verification_feedback"] = feedback
    final_answer = _check_and_rewrite_answer(
        state,
        answer,
        verification_feedback=feedback,
    )
    if not final_answer:
        raise RuntimeError("check and rewrite returned an empty answer")

    final_answer = _sanitize_final_answer(final_answer)
    state["answer"] = format_answer_with_image_list(
        final_answer, state.get("contexts", "")
    )
    return state


def _verify_answer_support(state: AgentState, answer: str) -> str:
    if not answer.strip():
        return "UNSUPPORTED: empty answer"
    prompt = ANSWER_VERIFICATION_PROMPT.format(**_prompt_inputs(state, answer=answer))
    return _invoke_chat(prompt, error_context="verify answer support")


def _check_and_rewrite_answer(
    state: AgentState,
    answer: str,
    verification_feedback: str = "",
) -> str:
    prompt = CHECK_AND_REWRITE_PROMPT.format(
        **_prompt_inputs(
            state,
            answer=answer,
            verification_feedback=verification_feedback,
        )
    )
    return _invoke_chat(prompt, error_context="check and rewrite answer")


def _prompt_inputs(
    state: AgentState,
    answer: str | None = None,
    verification_feedback: str = "",
) -> dict[str, str]:
    values = {
        "image_summary": state.get("image_summary", NO_IMAGE_SUMMARY),
        "contexts": state.get("contexts", ""),
        "response_plan": state.get("response_plan") or _build_response_plan(state),
        "verification_feedback": verification_feedback,
        "common_policy": COMMON_POLICY,
        "question": state["question"],
    }
    if answer is not None:
        values["answer"] = answer
    return values


def _sanitize_final_answer(answer: str) -> str:
    cleaned = answer.strip()
    cleaned = re.sub(r"^\s*(?:最终回复|终稿|答案|客服回复)\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"(?i)self-rag evidence verification feedback:.*", "", cleaned)
    cleaned = re.sub(r"根据(?:检索)?证据[，,：:]\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


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
        model = _init_openai_model(
            model=settings.chat_model,
            base_url=settings.openai_base_url,
            temperature=0.2,
            max_tokens=CHAT_MAX_TOKENS,
        )
        return _OUTPUT_PARSER.invoke(model.invoke([HumanMessage(content=prompt)])).strip()
    except Exception as exc:
        raise RuntimeError(f"failed to {error_context} with chat model") from exc


def _init_openai_model(
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int | None = None,
):
    settings = get_settings()
    kwargs = {
        "model": model,
        "model_provider": "openai",
        "api_key": settings.openai_api_key,
        "base_url": base_url,
        "temperature": temperature,
        "timeout": settings.model_timeout_seconds,
        "max_retries": 1,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)


def _image_summary_content(images: list[str]) -> list[dict]:
    content = [
        {
            "type": "text",
            "text": IMAGE_SUMMARY_PROMPT,
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image}} for image in images[:3]
    )
    return content


def _require_chat_model() -> None:
    if not get_settings().has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be configured so the chat model can generate answers."
        )
