# Family Heritage RAG Demo

一个可快速落地的“家族传承”Demo：
- 上传精英人物的故事、金句、经验
- 自动切片 + 向量化存储（SQLite）
- 用户可和指定人物进行检索增强对话（RAG）

> 已真实接入 OpenAI 大模型（Embedding + Chat）。

## 1. 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

打开：http://localhost:8000

## 2. 使用流程

1. 在“上传资料”里填写人物姓名、标题、内容。
2. 点击“上传并向量化”。
3. 在“开始对话”里输入同一人物姓名和问题。
4. 系统会先检索最相关片段，再调用模型生成回答。

## 3. API 一览

- `POST /api/upload`
  - form: `person`, `title`, `content`
- `POST /api/chat`
  - form: `person`, `question`

## 4. 下一步建议（你可以做的）

为了让这个 demo 快速升级为可演示产品，你可以配合我做这些：

1. **准备 3~5 位人物的高质量素材**
   - 每人 2000~5000 字：生平、关键决策、失败教训、金句。
2. **定义回答风格模板**
   - 例如：更理性 / 更鼓励 / 更直给。
3. **接入登录与权限**
   - 家族成员可见；或按家族分库。
4. **增加审核机制**
   - 上传后先审核，再进入知识库。
5. **如果要上生产**
   - SQLite -> pgvector / Milvus / Weaviate；
   - 增加缓存、埋点、对话历史、内容安全过滤。

## 5. 用 Codex Agent SDK 的升级方向

当前版本已可跑通真实 RAG。后续可引入 Agent SDK：
- 把“检索人物片段”封装为 tool
- 让 agent 自动追问（当资料不足时）
- 加入“生成人物年表/家风总结”的自动化任务

