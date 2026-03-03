import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import Chunk, app, chunk_text, top_k_context, validate_text_field


def test_chunk_text_overlap_behavior() -> None:
    text = "a" * 900
    chunks = list(chunk_text(text, chunk_size=360, overlap=80))
    assert len(chunks) == 3
    assert len(chunks[0]) == 360
    assert chunks[0][-80:] == chunks[1][:80]


def test_validate_text_field_rejects_empty() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_text_field("标题", "   ")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "标题不能为空"


def test_top_k_context_returns_highest_score(monkeypatch) -> None:
    chunks = [
        Chunk(id=1, person="A", title="t1", content="c1", embedding=np.array([1.0, 0.0], dtype=np.float32)),
        Chunk(id=2, person="A", title="t2", content="c2", embedding=np.array([0.0, 1.0], dtype=np.float32)),
    ]

    monkeypatch.setattr("app.main.embedding_for", lambda q, c: np.array([1.0, 0.0], dtype=np.float32))
    selected = top_k_context(chunks, "question", client=object(), k=1)

    assert len(selected) == 1
    assert selected[0].id == 1


def test_api_input_validation() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        data={"person": " ", "question": "你好"},
    )
    assert response.status_code == 400
    assert "人物姓名不能为空" in response.json()["detail"]
