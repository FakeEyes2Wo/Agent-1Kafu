from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .config import get_settings
from .graph import answer_question, response_payload


bearer_scheme = HTTPBearer(auto_error=False)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    images: list[str] = Field(default_factory=list)
    session_id: str | None = None
    stream: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Kefu Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(
        req: ChatRequest,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
        ],
    ) -> dict:
        settings = get_settings()
        if credentials is None or credentials.credentials != settings.kafu_api_token:
            raise HTTPException(status_code=401, detail="invalid token")

        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        answer, session_id = answer_question(question, req.images, req.session_id)
        return response_payload(answer, session_id)

    return app


app = create_app()
