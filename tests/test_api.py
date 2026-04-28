from fastapi.testclient import TestClient

from kefu_agent import api


def test_chat_requires_bearer_token():
    client = TestClient(api.create_app())

    response = client.post("/chat", json={"question": "hello"})

    assert response.status_code == 401


def test_chat_accepts_valid_bearer_token(monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda question, images, session_id: ("ok", "sid"))
    client = TestClient(api.create_app())

    response = client.post(
        "/chat",
        headers={"Authorization": "Bearer change-me"},
        json={"question": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "ok"
