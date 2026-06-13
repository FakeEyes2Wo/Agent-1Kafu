ZH_IMAGE_PARSE_PROMPT = """

你是一个说明书图片解析器。

当前图片没有在手册正文的 <PIC> 引用中出现。
但是我们根据图片名前缀和 image_name_counts.csv 推测它可能属于以下手册：

{manual_text}

图片文件名：
{img_name}

下面是推测所属手册的背景内容，只能用于理解产品类型，不代表该图片一定出现在某个具体步骤中：

{manual_background}

请生成 ImageSpec，要求如下：

1. imgname 必须严格等于：{img_name}
2. description 使用简体中文。
3. 主要依据图片本身描述，不要编造具体步骤位置。
4. 如果图片中有英文文字、按钮、标签、警告语，请保留英文原文并用中文解释。
5. 必须在 description 中说明：该图片未在正文 <PIC> 中出现，具体用途需要结合人工确认。
6. 如果图片信息有限，请明确说明“图片可见信息有限”。

description 建议包含：
- 图片可见内容；
- 可能对应的产品或部件；
- 可能用途，但必须使用“不确定”“可能”等措辞；
- 检索关键词。




"""

EN_IMAGE_PARSE_PROMPT = """You analyze images from product manuals.
Use the target image and its manual context to describe visible parts, labels, icons, states,
directions, and procedural relationships. Do not invent absent details. Return an ImageSpec:
use the provided file name as imgname and put only the image description in description."""


