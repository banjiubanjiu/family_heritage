from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "heritage.db"
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Family Heritage RAG Demo")


@dataclass
class Chunk:
    id: int
    person: str
    title: str
    content: str
    embedding: np.ndarray


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="请先配置 OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("CHAT_MODEL", "qwen-max").startswith("qwen"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )


def validate_text_field(field_name: str, value: str, max_len: int = 200) -> str:
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{field_name}不能为空")
    if len(normalized) > max_len:
        raise HTTPException(status_code=400, detail=f"{field_name}长度不能超过{max_len}字符")
    return normalized


def chunk_text(text: str, chunk_size: int = 360, overlap: int = 80) -> Iterable[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def embedding_for(text: str, client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        input=text,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def fetch_person_chunks(person: str) -> list[Chunk]:
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, person, title, content, embedding FROM memories WHERE person = ?",
            (person,),
        ).fetchall()
    chunks = []
    for row in rows:
        chunks.append(
            Chunk(
                id=row["id"],
                person=row["person"],
                title=row["title"],
                content=row["content"],
                embedding=np.array(json.loads(row["embedding"]), dtype=np.float32),
            )
        )
    return chunks


def top_k_context(chunks: list[Chunk], question: str, client: OpenAI, k: int = 4) -> list[Chunk]:
    if not chunks:
        return []
    q_emb = embedding_for(question, client)
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        return chunks[:k]

    scored = []
    for chunk in chunks:
        denom = np.linalg.norm(chunk.embedding) * q_norm
        score = 0.0 if denom == 0 else float(np.dot(chunk.embedding, q_emb) / denom)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
def upload_memory(
    person: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
) -> JSONResponse:
    client = get_client()

    person = validate_text_field("人物姓名", person, max_len=50)
    title = validate_text_field("标题", title, max_len=120)
    content = validate_text_field("内容", content, max_len=20000)

    rows = []
    for idx, chunk in enumerate(chunk_text(content)):
        emb = embedding_for(chunk, client)
        rows.append((person, f"{title} #{idx + 1}", chunk, json.dumps(emb.tolist())))

    with db_conn() as conn:
        conn.executemany(
            "INSERT INTO memories (person, title, content, embedding) VALUES (?, ?, ?, ?)",
            rows,
        )

    return JSONResponse({"ok": True, "inserted": len(rows)})


@app.post("/api/chat")
def chat_with_mentor(
    person: str = Form(...),
    question: str = Form(...),
) -> JSONResponse:
    client = get_client()
    person = validate_text_field("人物姓名", person, max_len=50)
    question = validate_text_field("问题", question, max_len=1000)

    chunks = fetch_person_chunks(person)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"没有找到 {person} 的资料，请先上传故事")

    selected = top_k_context(chunks, question, client)
    context_text = "\n\n".join([f"[{c.title}]\n{c.content}" for c in selected])

    system_prompt = (
        "你是一个家族传承助手。请严格根据提供的资料回答，"
        "尽量用第一人称呈现这位人物的经验与观点。"
        "如果资料里没有答案，要明确说资料未提及并给出可能的补充方向。"
    )

    completion = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen-max"),
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"人物：{person}\n\n"
                    f"资料片段：\n{context_text}\n\n"
                    f"用户问题：{question}"
                ),
            },
        ],
    )
    answer = completion.choices[0].message.content or ""

    return JSONResponse(
        {
            "ok": True,
            "answer": answer,
            "references": [
                {"id": c.id, "title": c.title, "content": c.content[:120]} for c in selected
            ],
        }
    )
