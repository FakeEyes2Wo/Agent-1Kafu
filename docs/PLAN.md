当前的代码设计


对于当前的代码设计，

对于下面的函数，如果名字不好，你可以替换成更好的名字。但不要增删其功能。

RAG part
我们首先需要的是关于图片 和图片语义的对应。

这里定义一个数据结构

```python  /src/kefu_agent/schemas.py
from pydantic import BaseModel
class ImageSpec(BaseModel):
    imgname: Field(str,description="图片文件名")
    description: Field(str,description="这个图片的具体内容")

    def get_prompt(self,id:int):
        这个函数用来获取<image></image>结尾的一段prompt
        这段prompt用来替换手册里面的<PIC>。
        "<image id='id'>中间是ImageSpec的Description</image>"
```
同时对于所有的文档，我们发现定义一个数据结构去处理它会更加方便

```python /src/kefu_agent/schemas.py
class ManualEntry():
    content:str=Field(description="当前表项的内容")
    img_list:List[str]
    def __init__(self,doc_str):
        doc_str是手册里面的一行[]包裹的文本。
        我们通过解析这个文本获取里面的content和img_list.

    def change_PIC(self,img_pairs)->str:
    这个函数的作用是匹配content所有的PIC部分，然后将其对应的file_list里面的文件名取出，
    然后用这个对应的文件名去提取img_pairs里面的ImageSpec，然后通过这里的ImageSpec.get_prompt()，获取到图片的
    文字描述，然后将文字描述替换进去。

def turn_description_PIC(content:str,img_pairs)->Tuple(content:str,file_list:List[str]):
    这个函数的作用是将change_PIC过后的文本以及大模型输出的文本，重新变成输出的content和对应的file_list.
    由于这个函数不仅仅要给FileContent用，所以这个函数不作为FileContent的类方法。
    这里我们的转化逻辑是通过get_prompt里面的<image id="id"> </image>里面的id实现的。
    首先我们匹配<image>标签，然后根据id，记录对应的filename.
    我们可以先匹配到所有的id，然后按照顺序生成file_list.
    然后将所有的<image></image>标签变成单独的<PIC>标签。

class ManualDocument():
    file_path:str = Field(description="这个文档的文档路径")
    file_name:str =Field(description="这个文档的文件名")
    content_list = List[FileContent]
    def __init__(self,file_path:str):
        这里通过输入文件路径去匹配所有的手册内容。



```
那么我们需要第二个处理的阶段。这个阶段放在init_rag函数里面


init_rag()介绍。


init_rag  首先我们通过vLLM去识别里面的所有图片，为每一个图片提供一个ImageSpec。
这里通过vLLM识别图片的过程应该是一套相对比较复杂的流程，
我们定义这个流程为

实际上我们需要一个函数去解析一个手册里面的
所有图片的出现  以及其出现的对应上下文。
这个函数需要同时对中英文手册起作用。
所以其输入应该是ManualEntry。同时有一个mode：literal["zh","en"]
# 这里我们简单切分即可。比方说按照行来切分，上下两行，同时满足<PIC>中心的前后200字等等。这种简单的得到每个图片的context
# en部分也是如此。

```python /src/kefu_agent/utils.py

PIC_RE = re.compile(r"<PIC>")


class ImageContext(BaseModel):
    imgname: str
    pic_indices: List[int]
    context: str




def get_line_range(text: str, pos: int, line_window: int = 2) -> tuple[int, int]:
    """
    根据字符位置 pos，返回其所在行上下 line_window 行的字符区间。
    """
    lines = text.splitlines(keepends=True)

    cursor = 0
    spans = []

    for line in lines:
        start, end = cursor, cursor + len(line)
        spans.append((start, end))
        cursor = end

    for i, (start, end) in enumerate(spans):
        if start <= pos < end:
            left = max(0, i - line_window)
            right = min(len(spans) - 1, i + line_window)
            return spans[left][0], spans[right][1]

    return 0, len(text)

def get_context_range(
    text: str,
    pos: int,
    line_window: int = 2,
    char_window: int = 200,
) -> tuple[int, int]:
    line_start, line_end = get_line_range(
        text=text,
        pos=pos,
        line_window=line_window,
    )

    char_start = max(0, pos - char_window)
    char_end = min(len(text), pos + char_window)

    start = min(line_start, char_start)
    end = max(line_end, char_end)

    return start, end
    
def render_pic_context(
    text: str,
    img_list: List[str],
    target_imgname: str,
    start: int,
    end: int,
) -> str:
    """
    渲染 text[start:end]。

    同一个 target_imgname 如果在窗口内出现多次，
    所有对应位置都会被标记为 TARGET_IMAGE。
    """
    matches = list(PIC_RE.finditer(text))

    parts = []
    cursor = start

    for pic_idx, match in enumerate(matches):
        m_start, m_end = match.span()

        if m_end <= start or m_start >= end:
            continue

        parts.append(text[cursor:m_start])

        imgname = img_list[pic_idx]
        tag = "TARGET_IMAGE" if imgname == target_imgname else "OTHER_IMAGE"

        parts.append(f"[{tag}: {imgname}, pic_idx={pic_idx}]")
        cursor = m_end

    parts.append(text[cursor:end])

    return "".join(parts).strip()

def build_image_context_index(
    entry: "ManualEntry",
    mode: Literal["zh", "en"] = "zh",
    line_window: int = 2,
    char_window: int = 200,
) -> Dict[str, ImageContext]:
    """
    构建：
    {
        imgname: ImageContext(
            imgname=...,
            pic_indices=[...],
            context=...
        )
    }
    """
    text = entry.content
    img_list = entry.img_list
    matches = list(PIC_RE.finditer(text))

    if len(matches) != len(img_list):
        raise ValueError(
            f"<PIC> 数量和 img_list 不一致: pic={len(matches)}, img={len(img_list)}"
        )

    img_to_indices: Dict[str, List[int]] = {}

    for pic_idx, imgname in enumerate(img_list):
        img_to_indices.setdefault(imgname, []).append(pic_idx)

    result: Dict[str, ImageContext] = {}

    header = (
        "以下是目标图片在手册中的上下文。[TARGET_IMAGE] 表示该图片的所有出现位置。"
        if mode == "zh"
        else "Below is the target image context. [TARGET_IMAGE] marks all occurrences of this image."
    )

    for imgname, pic_indices in img_to_indices.items():
        chunks = []
        seen = set()

        for pic_idx in pic_indices:
            pos = matches[pic_idx].start()

            start, end = get_context_range(
                text=text,
                pos=pos,
                line_window=line_window,
                char_window=char_window,
            )

            chunk = render_pic_context(
                text=text,
                img_list=img_list,
                target_imgname=imgname,
                start=start,
                end=end,
            )

            key = re.sub(r"\s+", " ", chunk).strip()

            if key and key not in seen:
                seen.add(key)
                chunks.append(chunk)

        context = header + "\n\n" + "\n\n---\n\n".join(chunks)

        result[imgname] = ImageContext(
            imgname=imgname,
            pic_indices=pic_indices,
            context=context,
        )

    return result


## Usage
ctx_index = build_image_context_index(
    entry=manual_entry,
    mode="zh",
    line_window=2,
    char_window=200,
)

ctx = ctx_index["Manual17_0"].context
```
那么我们应该将所有手册里面的上下文context提取出来过后，将所有dict合在一起，然后保存到rag_data/cache/img_context.csv里面，方便提取
我们记这个大dict为
img_context_dict.



```python  src/kefu_agent/preprocess.py
def build_image_specs():
    # 这里我们首先统计了每一个图片的出现手册名字和次数。   发现每一个图片只会在一个手册中出现，但是一个图片可能在手册中出现多次
    # 同时还有6个没有出现过的图片。
    # 我们的手册是有中英文的。  所以在处理的时候，我们对于中英文分别选取不同的处理管道
    
    img_pairs: Dict[str, ImageSpec] = {"图片名":对应的ImageSpec,...}
    我们首先通过ManualDocument 读取所有的文件，得到doc_list.
    对于除了"汇总英文手册.txt"其他的手册，我们首先统计这个手册里面
    出现了几张图片。
    IMAGE_PATH = r"data/手册/插图"
    for img_name in 手册的所有图片名:
        img = base64编码(IMAGE_PATH+"/"+img_name)
        img_context_dict[img_name]+img 送入vllm，   这里需要设计system_prompt  ZH_IMAGE_PARSE_PROMPT="""prompt"""
        这个vllm.with_structured_output(ImageSpec) 
        
        img_spec = vllm.invoke(所有文本内容+img)
        img_pairs[img_name]=img_spec

    对于英文手册:
    for img_name in 英文手册的所有图片名:
        img_context_dict[img_name]+img送入vllm， 这里需要设计system_prompt  设计为 EN_IMAGE_PARSE_PROMPT="""prompt"""


    对于没有出现在手册中的图片:
        unknown_imgs = ["Camera_20.png", "Camera_66.png", "Camera_67.png", "Dish_washer_04.png", "Dish_washer_05.png", "Dish_washer_06.png", "drill0_07.png", "drill0_13.png", "generator_02.png"]
        我们首先统计这些图片名的_之前的名称和其所在的手册。这里已经通过脚本"scripts/count_image_names.py"完成了。
        所以直接查阅"image_name_counts.csv"（这个在项目根目录下）
        for img_name in unknown_imgs:
            
            manual_name = image.split("_")[0]
            如果manual_name 相同的属于 中文手册，那么使用ZH_IMAGE_PARSE_PROMPT  否则EN_IMAGE_PARSE_PROMPT
            img = base64编码(IMAGE_PATH+"/"+img_name)
            img_spec = vllm.invoke(所有文本内容+img)
```


我们将这个每个图片的ImageSpec 存入一个List里面。这个List我们定义为 img_pairs。
img_pairs = [{"图片名":对应的ImageSpec,...}]

对于生成后的img_pairs， 我们将其每个图片名做一个自增的id 从1开始
img_label_pairs = {"id":"图片名"}

然后通过pandas 将这个List 转成DataFrame，再转换成csv文件,并保存到 rag_data目录。
这里的csv文件的表头为
id,imgname,description,


然后我们将手册里面所有的<PIC>全部替换掉。
我们注意到手册里面的格式是
["手册内容<PIC>手册内容",["文件名","文件名"]]
所以我们先读取手册的所有字符后，直接转List，然后提取
content以及file_list

这里的替换需要两个工具函数change_PIC，turn_description_PIC

我们对我们提取到的content 使用change_PIC函数，得到我们进行rag的原始文本。
这部分原始文本由于过多，所以我们的将其暂存到rag_data/cache目录下。
接下来进行rag

首先设计chunk的拆分模式
这次chunk拆分模式相对应该更细致一点。




在完成了以上的功能后，我们的init_rag部分就结束了

