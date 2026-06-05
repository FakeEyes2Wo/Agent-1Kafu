from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    manual: str
    title: str
    text: str
    image_ids: list[str]
    vector: list[float]
    manual_language: str = "zh"
    score: float = 0.0
    chunk_type: str = "text"
