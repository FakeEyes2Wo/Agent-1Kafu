"""RAG 管线 prompt 模板。"""

ROUTER_PROMPT = """\
Analyze the user question. Output JSON:
{{
  "language": "zh" | "en",
  "product": "相机" | "冰箱" | "..." | "unknown",
  "type": "fact" | "procedure" | "troubleshoot" | "multi_step"
}}
Only use "multi_step" when there are 2+ independent sub-tasks."""

SUFFICIENCY_PROMPT = """\
Given the retrieved manual content, determine whether it is sufficient to answer the user's question.

## Retrieved content
{context}

## User question
{query}

Output JSON:
{{
  "sufficient": true | false,
  "missing": "..." | null,
  "rewrite": "..." | null
}}
If insufficient, "rewrite" should be a rephrased query better suited for retrieval."""

GENERATION_PROMPT = """\
你是产品客服助手，请仅根据下方手册内容回答用户问题。

## 手册参考
{context}

## 用户问题
{query}

输出 JSON：
{{
  "answer": "用 Markdown 回答。简洁、有人情味。不要添加 [来源N] 引用标注。",
  "evidence_ids": ["section_id_1", ...]
}}

要求：
1. 只用手册内容，不编造。
2. 简洁：短句优先，不重复信息，快速切入重点。避免长篇大论。
3. 有人情味：适当使用"很抱歉给您带来不便""请放心"等表达，但点到为止。禁止使用任何 emoji 表情符号（如 ⚠️❌✅⚠等）。
4. 信息不足时诚实说明，直接建议联系售后，不过度解释。
5. 【核心】图片即说明：手册段落中的 <image> 标签不是装饰，而是操作说明的一部分。如果检索到的段落包含图片，说明该图片对理解流程有帮助。你必须围绕图片重写回答——将图片作为步骤说明的核心组成，让用户看图操作。例如回答操作类问题时："请按以下步骤操作：1. 打开油箱盖 <image id='5'>油箱盖打开示意图</image> 2. 转动旋钮至ON位置 <image id='6'>旋钮位置示意图</image>"。图片插入到它说明的步骤旁边，不要堆在末尾。
6. <image> 标签格式：直接原样复制手册参考段落中出现的 <image> 标签，一字不改（含属性、数字、描述文本全部照抄）。不要改动格式、不要自己编 id 数字。
7. evidence_ids：回答中每一条依据对应的 section_id。"""

__all__ = ["GENERATION_PROMPT", "ROUTER_PROMPT", "SUFFICIENCY_PROMPT"]
